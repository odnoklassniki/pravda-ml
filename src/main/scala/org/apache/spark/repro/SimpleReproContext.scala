package org.apache.spark.repro

import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

class SimpleReproContext private
(spark: SparkSession, basePath: String, tags: Seq[(String,String)]) extends ReproContext {

  def this(basePath: String)(implicit spark: SparkSession) = this(spark, basePath, Seq())

  var accumulatedMetrics : Seq[DataFrame] = Seq()

  var accumulatedParams: Seq[(Seq[String], Params)] = Seq()

  override def persistEstimator(estimator: MLWritable): Unit = {
    estimator.save(basePath + "/estimator")
  }

  override def persistModel(model: MLWritable): Unit = {
    model.save(basePath + "/model")
  }

  override def dive(tags: Seq[(String, String)]): ReproContext = new SimpleReproContext(
    spark, basePath, this.tags ++ tags)

  override def logParams(params: Params, path: Seq[String]): Unit =
    accumulatedParams = accumulatedParams :+ path -> params

  override def logMetircs(metrics: DataFrame): Unit = accumulatedMetrics = accumulatedMetrics :+ metrics

  override def start(): Unit = {
    import spark.implicits._
    accumulatedParams.foreach {
      case (path, params) => params.params.view
        .filter(params.isSet)
        .map(x => x.name -> x.asInstanceOf[Param[Any]].jsonEncode(params.get(x).get))
        .toDF("param", "value")
        .write.parquet(path.mkString(taggedPrefix + "/params/", "/", ""))
    }
  }

  override def finish(): Unit = {
    accumulatedMetrics.reduceOption(_ unionByName _).foreach(
      _.write.parquet(taggedPrefix + "/metrics"))
  }

  private def taggedPrefix: String = {
    tags.map(x => x._1 + "=" + x._2).mkString(basePath + "/", "/", "")
  }
}
