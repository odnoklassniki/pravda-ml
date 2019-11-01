package org.apache.spark.repro

import org.apache.spark.ml.param.{Param, ParamPair, Params}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession, functions}

class SimpleReproContext private
(spark: SparkSession, basePath: String, tags: Seq[(String,String)]) extends ReproContext {

  def this(basePath: String)(implicit spark: SparkSession) = this(spark, basePath, Seq())

  var accumulatedMetrics : Seq[DataFrame] = Seq()

  var accumulatedParams: Seq[(Seq[String], Iterable[ParamPair[_]])] = Seq()

  override def persistEstimator(estimator: MLWritable): Unit = {
    estimator.save(basePath + "/estimator")
  }

  override def persistModel(model: MLWritable): Unit = {
    model.save(basePath + "/model")
  }

  override def dive(tags: Seq[(String, String)]): ReproContext = new SimpleReproContext(
    spark, basePath, this.tags ++ tags)

  override def logParamPairs(params: Iterable[ParamPair[_]], path: Seq[String]): Unit =
    accumulatedParams = accumulatedParams :+ path -> params

  override def logMetircs(metrics: => DataFrame): Unit = accumulatedMetrics = accumulatedMetrics :+ metrics

  override def start(): Unit = {
    import spark.implicits._
    accumulatedParams.map {
      case (path, params) => params.view
        .map(x => x.param.name -> x.param.asInstanceOf[Param[Any]].jsonEncode(x.value))
        .toSeq
        .toDF("param", "value")
        .withColumn("path", functions.lit(path.mkString("/")))
    }.reduce(_ unionByName _)
      .write.parquet(taggedPrefix + "/params")
  }

  override def finish(): Unit = {
    accumulatedMetrics.reduceOption(_ unionByName _).foreach(
      _.write.parquet(taggedPrefix + "/metrics"))
  }

  private def taggedPrefix: String = {
    tags.map(x => x._1 + "=" + x._2).mkString(basePath + "/", "/", "")
  }
}
