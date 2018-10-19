package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.scala.Booster
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{BooleanParam, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasPredictionCol}
import org.apache.spark.sql.{Dataset, functions}


/**
  * USed to hack and extract some internals from dlmc XGBoost
  */
object XGBoostUtils {
  def getBooster(x: XGBoostClassificationModel): Booster = x._booster

  def getBooster(x: XGBoostRegressionModel): Booster = x._booster
}

trait OkXGBoostParams  extends HasFeaturesCol with HasPredictionCol {
  this: Params =>

  val densifyInput = new BooleanParam(this, "densifyInput",
    "In order to fix the difference between spark abd xgboost sparsity treatment")
  val predictAsDouble = new BooleanParam(this, "predictAsDouble",
    "Whenver to cast XGBoost prediction to double matching common behavior for other predictors.")
  val addRawTrees = new BooleanParam(this, "addRawTrees",
    "Whenever to add raw trees block to model summary.")
  val addSignificance = new BooleanParam(this, "addSignificance",
    "Whenever to add feature significance block to model summary.")

  def setAddSignificance(value: Boolean): this.type = set(addSignificance, value)

  def setAddRawTrees(value: Boolean): this.type = set(addRawTrees, value)

  def setDensifyInput(value: Boolean): this.type = set(densifyInput, value)

  def setPredictAsDouble(value: Boolean): this.type = set(predictAsDouble, value)

  protected def densifyIfNeeded(dataset: Dataset[_]) : Dataset[_] = {
    if ($(densifyInput)) {
      val densify = functions.udf((x: Vector) => x.toDense)
      val col = getFeaturesCol
      val metadata = dataset.schema(col).metadata

      dataset.withColumn(
        col,
        densify(dataset(col)).as(col, metadata))
    } else {
      dataset
    }
  }
}

trait OkXGBoostClassifierParams extends XGBoostClassifierParams with OkXGBoostParams

trait OkXGBoostRegressorParams extends XGBoostRegressorParams with OkXGBoostParams
