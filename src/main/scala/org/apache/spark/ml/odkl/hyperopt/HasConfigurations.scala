package org.apache.spark.ml.odkl.hyperopt
import org.apache.spark.ml.odkl.ModelWithSummary.Block

/**
  * Common summary block to store history of the hyperparameters search.
  */
trait HasConfigurations {
  val configurations: Block = Block("configurations")
}
