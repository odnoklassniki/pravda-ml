package org.apache.spark.ml.odkl

import org.apache.spark.ml.odkl.ModelWithSummary.Block

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains common summary blocks.
  */


/**
  * Block produced by a models with concept of feature weights (eg. linear models).
  */
trait HasWeights {
  val weights = Block("weights")

  val index: String = "index"
  val name: String = "name"
  val weight: String = "weight"
}

/**
  * Utility used for reporting single indexed feature weight.
  */
case class WeightedFeature(index: Int, name: String, weight: Double)

/**
  * Metrics block is added by the evaluators.
  */
trait HasMetricsBlock {
  val metrics = new Block("metrics")
}

/**
  * Block with information regarding features significance stat, produced during the
  * features selection stage.
  */
trait HasFeaturesSignificance {

  def featuresSignificance = Block("featuresSignificance")

  val feature_index: String = "index"
  val feature_name: String = "name"
  val avg: String = "avg"
  val std: String = "std"
  val n: String = "n"
  val significance: String = "significance"
}