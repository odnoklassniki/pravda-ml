package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utility used for linear scaling of features based on mean and variance.
  */

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.Estimator
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.ml.linalg.{BLAS, DenseVector, SparseVector, Vector}
import org.apache.spark.ml.odkl.Scaler.transformModel
import org.apache.spark.mllib
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, functions}
import org.apache.spark.mllib.linalg.VectorImplicits._

/**
  * Scaler parameters.
  */
trait ScalerParams extends Params with HasFeaturesCol {

  /**
    * Whether to center the data with mean before scaling.
    * It will build a dense output, so this does not work on sparse input
    * and will raise an exception.
    * Default: false
    *
    * @group param
    */
  val withMean: BooleanParam = new BooleanParam(this, "withMean",
    "Whether to center data with mean")

  /** @group getParam */
  def getWithMean: Boolean = $(withMean)

  def setWithMean(value: Boolean): this.type = set(withMean, value)

  /**
    * Whether to scale the data to unit standard deviation.
    * Default: true
    *
    * @group param
    */
  val withStd: BooleanParam = new BooleanParam(this, "withStd",
    "Whether to scale the data to unit standard deviation")

  /** @group getParam */
  def getWithStd: Boolean = $(withStd)

  def setWithStd(value: Boolean): this.type = set(withStd, value)

  setDefault(withMean -> true, withStd -> true)
}

/**
  * This is a specific implementation of the scaler for linear models. Uses the ability to propagate scaling to the
  * weights to avoid overhead when predicting.
  */
abstract class ScalerEstimator[M <: ModelWithSummary[M]]
(
  override val uid: String = Identifiable.randomUID("scalerEstimator"))
  extends Estimator[Scaler.Unscaler[M]] with ScalerParams with DefaultParamsWritable {

  protected def unscaleModel(model: M, scaler: StandardScalerModel): M

  override def fit(dataset: Dataset[_]): Scaler.Unscaler[M] = {
    val scaler = new StandardScaler($(withMean), $(withStd)).fit(
      dataset.select($(featuresCol)).rdd.map(r => mllib.linalg.Vectors.fromML(r.getAs[Vector](0))))

    new Scaler.Unscaler[M](scaler, unscaleModel).setParent(this)
  }

  override def transformSchema(schema: StructType): StructType = schema
}

class LinearScaleEstimator[M <: LinearModel[M]] extends ScalerEstimator[M] {
  override protected def unscaleModel(model: M, scaler: StandardScalerModel): M
  = Scaler.transformModel(model, scaler)

  override def copy(extra: ParamMap) = defaultCopy(extra)
}

object LinearScaleEstimator extends DefaultParamsReadable[LinearScaleEstimator[_]]

class CompositScaleEstimator[M <: LinearModel[M], C <: CombinedModel[M, C]] extends ScalerEstimator[C] {
  override protected def unscaleModel(model: C, scaler: StandardScalerModel): C = {
    model.transformNested(x => transformModel[M](x, scaler))
  }

  override def copy(extra: ParamMap) = defaultCopy(extra)
}

object CompositScaleEstimator extends DefaultParamsReadable[CompositScaleEstimator[_,_]]

object Scaler extends Serializable {
  /**
    * Given a linear model estimator first scale the data based on mean/variance, then train the model
    * and unscale its weights.
    *
    * @param estimator Nested linear model estimator.
    * @param scaler    Pre configured scaler (by default with mean and sd).
    * @return Linear model as produced by the nested estimator with unscaled weights.
    */
  def scale[M <: LinearModel[M]]
  (
    estimator: SummarizableEstimator[M],
    scaler: ScalerEstimator[M] = new LinearScaleEstimator[M]())
  (implicit m: Manifest[M])
  : UnwrappedStage[M, Unscaler[M]] = {

    new UnwrappedStage[M, Unscaler[M]](estimator, scaler)
  }

  /**
    * Extention for scaling composite models assuming their are composites of linear models.
    *
    * @param estimator Nested composite estimator.
    * @param scaler Scaler with te settings.
    * @return Composite model as produced by the nested estimator with all parts unscaled.
    */
  def scaleComposite[M <: LinearModel[M], C <: CombinedModel[M, C]]
  (
    estimator: SummarizableEstimator[C],
    scaler: ScalerEstimator[C] = new CompositScaleEstimator[M, C]()) : UnwrappedStage[C, Unscaler[C]] = {
    new UnwrappedStage[C, Unscaler[C]](estimator, scaler)
  }


  private def transformData(dataset: DataFrame, featuresCol: String, scaler: StandardScalerModel): DataFrame = {

    val structField = dataset.schema.fields(dataset.schema.fieldIndex(featuresCol))

    val transform = SparkSqlUtils.reflectionLock.synchronized(
      if (scaler.withMean) {
        // Only dense vector can be scaled with mean.
        functions.udf[Vector, Vector](x => scaler.transform(x.toDense))
      }
      else {
        functions.udf[Vector, Vector](x => scaler.transform(x))
      })

    dataset.withColumn(
      featuresCol, transform(dataset(featuresCol)).as(featuresCol, structField.metadata))
  }

  private[odkl] def transformModel[M <: LinearModel[M]](model: M, scaler: StandardScalerModel): M = {
    val nestedSummary: ModelSummary = model.summary

    val (coefficientsAndIntercept, delta) = if (scaler.withMean) {
      val noMeanModel = scaler.setWithMean(false)

      val newCoefficients = noMeanModel.transform(model.getCoefficients)
      val interceptDelta = BLAS.dot(newCoefficients, scaler.mean)

      (newCoefficients, model.getIntercept - interceptDelta)  -> interceptDelta

    } else {
      (scaler.transform(model.getCoefficients), model.getIntercept) -> 0.0
    }

    val summary = SparkSqlUtils.reflectionLock.synchronized(
      nestedSummary.transform(model.weights -> (data => {
        val mean = functions.udf[Double, Int](i => if (i >= 0) scaler.mean(i) else 1.0)
        val std = functions.udf[Double, Int](i => if (i >= 0) scaler.std(i) else 0.0)
        val weight = functions.udf[Double, Int, Double]((i, w) => if (i >= 0) w / scaler.std(i) else
          // Only for final model intercept we can provide true unscaled variant by a simple UDF.
          if (w == model.getIntercept) coefficientsAndIntercept._2 else Double.NaN)

        // TODO: Configure column names
        data
            .withColumns(
              Seq(s"unscaled_${model.weight}", model.weight, "value_mean", "value_std"),
              Seq(
                data(model.weight),
                weight(data(model.index), data(model.weight)),
                mean(data(model.index)),
                std(data(model.index))
              )
            )
      }))
    )

    model.copy(
      summary,
      ParamMap(
        model.coefficients -> coefficientsAndIntercept._1,
        model.intercept -> coefficientsAndIntercept._2))
  }

  /**
    * Applies unscaling to the linear model weights.
    */
  class Unscaler[M <: ModelWithSummary[M]]
  (
    val scaler: StandardScalerModel,
    val modelTransformer : (M, StandardScalerModel) => M,
    override val uid: String = Identifiable.randomUID("unscaler")
  )
    extends ModelTransformer[M, Unscaler[M]] with HasFeaturesCol with ScalerParams {

    override def transform(dataset: Dataset[_]): DataFrame = Scaler.transformData(
      dataset.toDF, $(featuresCol), scaler)

    override def transformModel(model: M, originalData: DataFrame): M = modelTransformer(model, scaler)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema

    override def copy(extra: ParamMap): Unscaler[M] = copyValues(new Unscaler[M](scaler, modelTransformer), extra)
  }

}