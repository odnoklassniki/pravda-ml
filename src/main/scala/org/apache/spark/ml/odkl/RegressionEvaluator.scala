package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains classes supporting evaluation of regression.
  */

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types.{DoubleType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.util.Try

/**
  * Simple evaluator based on the mllib.RegressionMetrics.
  *
  * TODO: Add unit tests
  */
class RegressionEvaluator(override val uid: String) extends Evaluator[RegressionEvaluator](uid) {

  val throughOrigin = new BooleanParam(this, "throughOrigin",
    "True if the regression is through the origin. For example, in " +
      "linear regression, it will be true without fitting intercept.")

  def setThroughOrigin(value: Boolean): this.type = set(throughOrigin, value)

  def getThroughOrigin: Boolean = $(throughOrigin)

  def this() = this(Identifiable.randomUID("regressionEvaluator"))


  override def transform(dataset: Dataset[_]): DataFrame = {

    try {
      val predictions: RDD[(Double, Double)] = dataset.select($(predictionCol), $(labelCol))
        .rdd.map { case Row(score: Double, label: Double) => (score, label) }

      val metrics = Try(new RegressionMetrics(predictions))


      val rows = metrics.toOption.map(m => Seq(
        "r2" -> m.r2,
        "rmse" -> m.rootMeanSquaredError,
        "explainedVariance" -> m.explainedVariance,
        "meanAbsoluteError" -> m.meanAbsoluteError,
        "meanSquaredError" -> m.meanSquaredError
      ).map(Row.fromTuple)).getOrElse(Seq())

      SparkSqlUtils.reflectionLock.synchronized(
        dataset.sqlContext.createDataFrame(
          dataset.sparkSession.sparkContext.parallelize(rows, 1), transformSchema(dataset.schema)))
    } catch {
      // Most probably evaluation dataset is empty
      case e: Exception =>
        logWarning("Failed to calculate metrics due to " + e.getMessage)
        SparkSqlUtils.reflectionLock.synchronized(
          dataset.sqlContext.createDataFrame(
            dataset.sparkSession.sparkContext.emptyRDD[Row], transformSchema(dataset.schema)))
    }
  }

  override def copy(extra: ParamMap): RegressionEvaluator = {
    copyValues(new RegressionEvaluator(), extra)
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    new StructType()
      .add("metric", StringType, nullable = false)
      .add("value", DoubleType, nullable = false)
  }
}
