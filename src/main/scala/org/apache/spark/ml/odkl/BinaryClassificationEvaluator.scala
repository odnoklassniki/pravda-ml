package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains classes supporting evaluation of binary classifiers.
  */

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types.{DoubleType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.ml.linalg.{Vector, VectorUDT}

/**
  * Simple evaluator based on the mllib.BinaryClassificationMetrics.
  */

class BinaryClassificationEvaluator(override val uid: String) extends Evaluator[BinaryClassificationEvaluator](uid) {

  def this() = this(Identifiable.randomUID("binaryClassificationEvaluator"))

  val predictionIndex = new IntParam(this, "predictionIndex",
    "Select value by that index from probability vector.")

  setDefault(predictionIndex, 1)

  def setPredictionIndex(value: Int): this.type = set(predictionIndex, value)
  def getPredictionIndex = $(predictionIndex)

  val fmeasureThresholds: Param[Map[String, Double]] = JacksonParam.mapParam[Double](
    this, "defaultValues", "Default values to assign to columns")

  def setFmeasureThresholds(value: Map[String, Double]): this.type = set(fmeasureThresholds, value)

  def getFmeasureThresholds = $(fmeasureThresholds)

  setDefault(fmeasureThresholds, Map[String, Double]("f1" -> 1.0))

  final val numBins:  IntParam = new  IntParam(
    this, "numBins", "How many points to add to nested curves (recall/precision or roc)")

  setDefault(numBins, 100)

  override def transform(dataset: Dataset[_]): DataFrame = {

    val predictions: RDD[(Double, Double)] = dataset.schema($(predictionCol)).dataType match {
      case _: VectorUDT => dataset.select($(predictionCol), $(labelCol))
        .rdd.map{case Row(score: Vector, label: Double) => (score($(predictionIndex)), label)}
      case _ => dataset.select($(predictionCol), $(labelCol))
        .rdd.map{case Row(score: Double, label: Double) => (score, label)}
    }

    val metrics = new BinaryClassificationMetrics(predictions, 100)

    val rows = if (metrics.roc().count() > 2) {
      dataset.sqlContext.sparkContext.parallelize(Seq(
        Seq[Any]("auc", metrics.areaUnderROC(), null, null),
        Seq[Any]("auc_pr", metrics.areaUnderPR(), null, null)
      ) ++
        $(fmeasureThresholds).map{case (name, threshold) => metrics.fMeasureByThreshold(threshold)
          .map(x => Seq[Any](name, x._2, "threshold", x._1))}.reduce(_ union _)
          .union(metrics.precisionByThreshold().map(x => Seq[Any]("precision", x._2, "threshold", x._1)))
          .union(metrics.recallByThreshold().map(x => Seq[Any]("recall", x._2, "threshold", x._1)))
          .union(metrics.pr().map(x => Seq[Any]("precision", x._2, "recall", x._1)))
          .union(metrics.roc().map(x => Seq[Any]("tp_rate", x._2, "fp_rate", x._1)))
          .collect()
        , 1)
        .map(x => Row.fromSeq(x))
    } else {
      dataset.sqlContext.sparkContext.parallelize(Seq[Row](), 1)
    }

    metrics.unpersist()

    SparkSqlUtils.reflectionLock.synchronized(
      dataset.sqlContext.createDataFrame(
        rows, transformSchema(dataset.schema)))
  }

  override def copy(extra: ParamMap): BinaryClassificationEvaluator = {
    copyValues(new BinaryClassificationEvaluator(), extra)
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    new StructType()
      .add("metric", StringType, nullable = false)
      .add("value", DoubleType, nullable = false)
      .add("x-metric", StringType, nullable = true)
      .add("x-value", DoubleType, nullable = true)
  }
}
