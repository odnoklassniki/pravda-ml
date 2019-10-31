package org.apache.spark.repro

import odkl.analysis.spark.util.Logging
import org.apache.spark.ml.odkl.HasMetricsBlock
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.joda.time.LocalDate
import org.mlflow.tracking.creds.MlflowHostCredsProvider
import org.mlflow.tracking.{ActiveRun, MlflowClient, MlflowContext}

import scala.collection.JavaConverters._

trait MLFlowClient {
  self =>
  val context: MlflowContext
  val run: ActiveRun

  def dive(newRun: String): MLFlowClient = {
    new MLFlowClient {
      override val context: MlflowContext = self.context
      override lazy val run: ActiveRun = self.context.startRun(newRun, self.run.getId)
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
}

class MlFlowReproContext private[repro]
(spark: SparkSession,
 basePath: String,
 mlFlow: MLFlowClient,
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

  override def logParams(params: Params, path: Seq[String]): Unit = {
    val javaParams = params.params.view
      .filter(params.isSet)
      .map(x => (path :+ x.name).mkString("/") -> x.asInstanceOf[Param[Any]].jsonEncode(params.get(x).get))
      .toMap.asJava
    mlFlow.run.logParams(javaParams)
  }

  override def logMetircs(metrics: => DataFrame): Unit = {
    try {
      val groundMetrics: DataFrame = metrics
      val fields = groundMetrics.schema.fieldNames.toSet
      import groundMetrics.sqlContext.implicits._
      logInfo(s"Got metrics dataframe to log with fields $fields")

      val defaultFold = if (fields.contains("foldNum")) {
        logInfo("Filtering by fold")
        groundMetrics.filter('foldNum === -1)
      } else {
        groundMetrics
      }

      val scalarMetrics = if (fields.contains("x-value")) {
        logInfo("Filtering scalar metrics")
        defaultFold.filter($"x-value".isNull)
      } else {
        defaultFold
      }

      val metricsMap: Map[String, java.lang.Double] = if (Set("metric", "value", "isTest").subsetOf(fields)) {
        logInfo("Logging train/test metrics")
        extractMetricValue(scalarMetrics)
          .select("metric", "value", "isTest")
          .map {
            case Row(metric: String, value: Double, isTest: Boolean)
            => s"$metric on ${if (isTest) "test" else "train"}" -> java.lang.Double.valueOf(value)
          }
          .collect()
          .toMap
      } else if (Set("metric", "value").subsetOf(fields)) {
        logInfo("Logging plain metrics")
        extractMetricValue(scalarMetrics)
          .select("metric", "value")
          .map {
            case Row(metric: String, value: Double) => metric -> java.lang.Double.valueOf(value)
          }
          .collect()
          .toMap
      } else {
        logWarning("Unknown schema, not logging the metrics.")
        Map[String, java.lang.Double]()
      }

      if (metricsMap.nonEmpty) {
        mlFlow.run.logMetrics(metricsMap.asJava)
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
