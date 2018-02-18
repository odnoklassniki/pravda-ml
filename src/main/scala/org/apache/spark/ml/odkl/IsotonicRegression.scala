package org.apache.spark.ml.odkl

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.regression.IsotonicRegressionModel
import org.apache.spark.ml.util._
import org.apache.spark.mllib.odkl.{IsotonicRegression => MLlibIsotonicRegression}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.storage.StorageLevel

/**
  * :: Experimental ::
  * Isotonic regression.
  *
  * Currently implemented using parallelized pool adjacent violators algorithm.
  * Only univariate (single feature) algorithm supported.
  *
  * Uses [[org.apache.spark.mllib.regression.IsotonicRegression]].
  *
  * ODKL Patch: Used to inject our patched mllib implementation.
  */
@Since("1.5.0")
@Experimental
class IsotonicRegression @Since("1.5.0")(@Since("1.5.0") override val uid: String)
  extends org.apache.spark.ml.regression.IsotonicRegression(uid) {

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("isoReg"))

  @Since("1.5.0")
  override def fit(dataset: Dataset[_]): IsotonicRegressionModel = {
    validateAndTransformSchema(dataset.schema, fitting = true)
    // Extract columns from data.  If dataset is persisted, do not persist oldDataset.
    val instances = extractWeightedLabeledPoints(dataset)
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val isotonicRegression = new MLlibIsotonicRegression().setIsotonic($(isotonic))
    val oldModel = isotonicRegression.run(instances)

    copyValues(new IsotonicRegressionModel(uid, oldModel).setParent(this))
  }
}

@Since("1.6.0")
object IsotonicRegression extends DefaultParamsReadable[IsotonicRegression] {

  @Since("1.6.0")
  override def load(path: String): IsotonicRegression = super.load(path)
}

