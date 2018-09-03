package ml.dmlc.xgboost4j.scala.spark

import ml.dmlc.xgboost4j.scala.Booster


/**
  * USed to hack and extract some internals from dlmc XGBoost
  */
object XGBoostUtils {
  def getBooster(x: XGBoostClassificationModel): Booster = x._booster
}

trait OkXGBoostClassifierParams extends XGBoostClassifierParams {

}
