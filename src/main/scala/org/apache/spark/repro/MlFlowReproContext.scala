package org.apache.spark.repro

import java.util

import odkl.analysis.spark.util.Logging
import org.apache.spark.ml.odkl.HasMetricsBlock
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.joda.time.LocalDate
import org.mlflow.tracking.creds.MlflowHostCredsProvider
import org.mlflow.tracking.{ActiveRun, MlflowClient, MlflowContext}

import scala.collection.JavaConverters._

trait MLFlowClient { self =>
  val context: MlflowContext
  val run: ActiveRun

  def dive(newRun: String) : MLFlowClient = {
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
    val newRun = mlFlow.dive(s"Fork for ${tags.map(_.productIterator.mkString("=")).mkString(", ")}")

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

  override def logMetircs(metrics: DataFrame): Unit = {
    val fields = metrics.schema.fieldNames.toSet
    import metrics.sqlContext.implicits._

    val javaMetrics: util.Map[String, java.lang.Double] = if (Set("metric", "value", "isTest").subsetOf(fields)) {
      metrics
        .select('metric.as[String], 'value.as[Option[Number]], 'isTest.as[Boolean])
        .filter(_._2.isDefined)
        .map { case (metric, value, isTest) => (s"$metric on (${if (isTest) "test" else "train"})"
          -> java.lang.Double.valueOf(value.get.doubleValue()))
        }
        .collect()
        .toMap
        .asJava
    } else if (Set("metric,value").subsetOf(fields)) {
      metrics
        .select('metric.as[String], 'value.as[Option[Number]])
        .filter(_._2.isDefined)
        .map { case (metric, value) => metric -> java.lang.Double.valueOf(value.get.doubleValue()) }
        .collect()
        .toMap
        .asJava
    } else {
      Map[String, java.lang.Double]().asJava
    }

    mlFlow.run.logMetrics(javaMetrics)
  }

  override def start(): Unit = {}

  override def finish(): Unit = mlFlow.run.endRun()


}
