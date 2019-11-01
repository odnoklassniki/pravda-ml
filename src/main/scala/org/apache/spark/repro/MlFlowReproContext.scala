package org.apache.spark.repro

import java.util.Collections

import odkl.analysis.spark.util.Logging
import org.apache.spark.ml.odkl.HasMetricsBlock
import org.apache.spark.ml.param.{Param, ParamPair, Params}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.{DataFrame, Row, SparkSession, functions}
import org.joda.time.LocalDate
import org.mlflow.api.proto.Service
import org.mlflow.tracking.creds.MlflowHostCredsProvider
import org.mlflow.tracking.{ActiveRun, MlflowClient, MlflowContext}

import scala.collection.JavaConverters._

trait MLFlowClient {
  self =>
  val context: MlflowContext
  val run: ActiveRun

  def logMetricsBatch(metrics: Iterable[Service.Metric] ) = {
    context.getClient.logBatch(run.getId,
      metrics.asJava,
      Collections.emptyList[Service.Param],
      Collections.emptyList[Service.RunTag])
  }

  def dive(newRun: String): MLFlowClient = {
    new MLFlowClient {
      override val context: MlflowContext = self.context
      override lazy val run: ActiveRun = {
        val run = self.context.startRun(newRun, self.run.getId)
        // Used to simplify navigation in ML Flow UI
        run.setTag("parent", self.run.getId)
        run
      }
    }
  }
}

class MLFlowRootClient private[repro](client: MlflowClient, experiment: String) extends MLFlowClient {
  override lazy val context: MlflowContext = {

    val experimentId = Option(client
      .getExperimentByName(experiment)
      .orElse(null))
      .map(_.getExperimentId)
      .getOrElse(client.createExperiment(experiment))

    new MlflowContext(client).setExperimentId(experimentId)
  }

  override lazy val run: ActiveRun = context.startRun("root")

  run.setTag("parent", "root")
}

case class MetricInfo(metric: String, value: Double, isTest: Option[Boolean], step: Long)

class MlFlowReproContext private[repro]
(spark: SparkSession,
 basePath: String,
 val mlFlow: MLFlowClient,
 tags: Seq[(String, String)]) extends ReproContext with Logging with HasMetricsBlock {

  private val startTime = new LocalDate()

  def this(creds: MlflowHostCredsProvider, basePath: String, experiment: String)(implicit spark: SparkSession)
  = this(spark, basePath, new MLFlowRootClient(new MlflowClient(creds), experiment), Seq())


  def this(url: String, basePath: String, experiment: String)(implicit spark: SparkSession)
  = this(spark, basePath, new MLFlowRootClient(new MlflowClient(url), experiment), Seq())

  override def persistEstimator(estimator: MLWritable): Unit = {
    //estimator.save(basePath + s"/experiment=${mlFlow.context.getExperimentId}/estimator")
  }

  override def persistModel(model: MLWritable): Unit = {
    //model.save(basePath + "/model")
  }

  override def dive(tags: Seq[(String, String)]): ReproContext = {
    val newRun: MLFlowClient = mlFlow.dive(s"Fork for ${tags.map(_.productIterator.mkString("=")).mkString(", ")}")

    newRun.run.setTags(tags.toMap.asJava)

    new MlFlowReproContext(spark, basePath, newRun, tags)
  }

  override def logParamPairs(params: Iterable[ParamPair[_]], path: Seq[String]): Unit = {
    val javaParams = params
      .map(x => (path :+ x.param.name).mkString("/") -> x.param.asInstanceOf[Param[Any]].jsonEncode(x.value))
      .toMap.asJava
    mlFlow.run.logParams(javaParams)
  }

  override def logMetircs(metrics: => DataFrame): Unit = {
    try {
      val groundMetrics: DataFrame = metrics
      val fields = groundMetrics.schema.fieldNames.toSet

      if (Set("metric", "value").subsetOf(fields)) {
        import groundMetrics.sqlContext.implicits._
        logInfo(s"Got metrics dataframe to log with fields $fields")

        val scalarMetrics = if (fields.contains("x-value")) {
          logInfo("Filtering scalar metrics")
          groundMetrics.filter($"x-value".isNull)
        } else {
          groundMetrics
        }

        val (withStep, converter) = if (fields.contains("invertedStep")) {
          logInfo("Adding step from inverted index.")
          scalarMetrics.withColumn("step", 'invertedStep.cast("long")) -> ((i: Long, max: Long) => max - i)
        } else if (fields.contains("step")) {
          scalarMetrics.withColumn("step", 'step.cast("long")) -> ((i: Long, max: Long) => i)
        } else {
          scalarMetrics.withColumn("step", functions.lit(0L).cast("long")) -> ((i: Long, max: Long) => i)
        }

        val withIsTest = if (fields.contains("isTest")) {
          withStep
        } else {
          withStep.withColumn("isTest", functions.lit(null).cast("boolean"))
        }

        val rows: Array[MetricInfo] = withIsTest
          .withColumn("value", 'value.cast("double"))
          .filter('value.isNotNull)
          .select(SparkSqlUtils.toStruct[MetricInfo]())
          .collect()


        if (rows.nonEmpty) {
          val maxStep: Long = rows.view.map(_.step).max

          val metricsBatch: Array[Service.Metric] = rows
            .map(x => Service.Metric.newBuilder()
              .setKey(x.isTest.map(y => s"${x.metric} on ${if (y) "test" else "train"}").getOrElse(x.metric))
              .setValue(x.value)
              .setStep(converter(x.step, maxStep))
              .build()
            )

          mlFlow.logMetricsBatch(metricsBatch)
        } else {
          logWarning("Got empty metrics set.")
        }
      } else {
        logWarning("Missing required fields for logging metrics to MLFlow - metric and value")
      }
    } catch {
      case e: Throwable =>
        logError("Exception while logging metrics", e)
        throw e
    }
  }

  private def extractMetricValue(scalarMetrics: DataFrame): DataFrame = {
    import scalarMetrics.sqlContext.implicits._
    scalarMetrics
      .withColumn("value", 'value.cast("double"))
      .filter('value.isNotNull)
  }

  override def start(): Unit = {}

  override def finish(): Unit = mlFlow.run.endRun()


}
