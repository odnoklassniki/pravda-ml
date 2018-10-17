package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.scala.Booster
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasPredictionCol}


/**
  * USed to hack and extract some internals from dlmc XGBoost
  */
object XGBoostUtils {
  def getBooster(x: XGBoostClassificationModel): Booster = x._booster
}

trait OkXGBoostClassifierParams extends XGBoostClassifierParams with HasFeaturesCol with HasPredictionCol  {
  val densifyInput = new BooleanParam(this, "densifyInput",
    "In order to fix the difference between spark abd xgboost sparsity treatment")

  val predictAsDouble = new BooleanParam(this, "predictAsDouble",
    "Whenver to cast XGBoost prediction to double matching common behavior for other predictors.")

  def getDensifyInput : Boolean = $(densifyInput)
}
