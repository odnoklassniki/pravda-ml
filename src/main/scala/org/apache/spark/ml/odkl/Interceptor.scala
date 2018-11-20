package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utility used to add intercept for the models.
  */

import org.apache.spark.ml._
import org.apache.spark.ml.attribute.{AttributeGroup, NumericAttribute}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, functions}

/**
  * Adds extra column to features vector with a fixed value of 1. Can be used with any model.
  */
class Interceptor(override val uid: String = Identifiable.randomUID("interceptor"))
  extends Transformer with HasFeaturesCol with DefaultParamsWritable {

  override def transform(dataset: Dataset[_]): DataFrame = intercept(dataset.toDF, $(featuresCol))

  def setFeaturesCol(value: String) : this.type = set(featuresCol, value)

  override def copy(extra: ParamMap): Interceptor = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  def intercept(dataset: DataFrame, column: String): DataFrame = {
    val structField: StructField = dataset.schema.fields(dataset.schema.fieldIndex(column))
    val attributeGroup = AttributeGroup.fromStructField(structField)

    val updatedGroup = attributeGroup.attributes.map(x => {
      new AttributeGroup(
        attributeGroup.name,
        x :+ NumericAttribute.defaultAttr
          .withIndex(attributeGroup.size)
          .withName("#intercept")
          .withMax(1.0)
          .withMin(1.0)
          .withStd(0.0))
    }).getOrElse(new AttributeGroup(attributeGroup.name, attributeGroup.size + 1))

    val addIntercept = functions.udf[Vector, Vector](x => Interceptor.appendBias(x))
    dataset.withColumn(
      column,
      addIntercept(dataset(column))
        .as(column, updatedGroup.toMetadata(structField.metadata)))
  }
}

object Interceptor extends DefaultParamsReadable[Interceptor] with Serializable {

  /**
    * Adds an intercept/unintercept stage to the estimator of a linear model.
    *
    * @param estimator Nested estimator with linear model as result.
    * @return Estimator which produce a linear model with explicit intercept.
    */
  def intercept[M <: LinearModel[M]](estimator: SummarizableEstimator[M], features: String = "features"):
  UnwrappedStage[M, Uninterceptor[M]] = {
    new UnwrappedStage[M, Uninterceptor[M]](
      estimator,
      new Uninterceptor[M](new Interceptor().setFeaturesCol(estimator match {
        case withCol : HasFeaturesCol => withCol.getFeaturesCol
        case _ => features
      })))
  }

  /**
    * Adds an intercept/unintercept stage to the composite estimator with nested linear models.
    *
    * @param estimator Nested composite estimator built of with linear model as result.
    * @return Estimator which produce a combination of linear models with explicit intercept.
    */
  def interceptComposite[M <: LinearModel[M], C <: CombinedModel[M,C]](estimator: SummarizableEstimator[C], features: String = "features"):
  UnwrappedStage[C, UninterceptorComposite[M,C]] = {
    new UnwrappedStage[C, UninterceptorComposite[M,C]](
      estimator,
      new UninterceptorComposite[M,C](new Interceptor().setFeaturesCol(estimator match {
        case withCol : HasFeaturesCol => withCol.getFeaturesCol
        case _ => features
      })))
  }


  /**
    * Adds 1 to the end of a vector.
    */
  def appendBias(vector: Vector): Vector = vector match {
    case dense: DenseVector => Vectors.dense(dense.toArray :+ 1.0)
    case sparse: SparseVector => Vectors.sparse(
      sparse.size + 1,
      sparse.indices :+ sparse.size,
      sparse.values :+ 1.0)
  }

  /**
    * Removes last column of a vector.
    */
  def removeBias(vector: Vector): Vector = vector match {
    case dense: DenseVector => Vectors.dense(dense.toArray.dropRight(1))
    case sparse: SparseVector => if (sparse.indices.length > 0 && sparse.indices.last == sparse.size - 1) {
      Vectors.sparse(
        sparse.size - 1,
        sparse.indices.dropRight(1),
        sparse.values.dropRight(1))
    } else {
      Vectors.sparse(
        sparse.size - 1,
        sparse.indices,
        sparse.values)
    }
  }

  def transformModel[M <: LinearModel[M]](model: M): M = {
    val coefficients: Vector = model.getCoefficients
    val interceptIndex: Int = coefficients.size - 1
    val summary = model.summary
    SparkSqlUtils.reflectionLock.synchronized(
      model.copy(
        summary.transform(model.weights -> (x => {
          val reindex = functions.udf[Int, Int](n => if (n == interceptIndex) -1 else n)

          x.withColumn(model.index, reindex(x(model.index)))
        })).blocks,
        ParamMap(
          model.coefficients -> Interceptor.removeBias(coefficients),
          model.intercept -> coefficients(interceptIndex)
        ))
    )
  }

  /**
    * Model transformer which moves weight trained for an intercept into the intercept parameter of the model (so
    * far supported only for linear models). This helps to avoid extra overhead when predicting.
    */
  class Uninterceptor[M <: LinearModel[M]](val interceptor: Interceptor)
    extends UnwrappedStage.PredefinedDataTransformer[M, Uninterceptor[M]](Identifiable.randomUID("uninterceptor"), interceptor)
  {

    override def transformModel(model: M, originalData: DataFrame): M = Interceptor.transformModel(model)

    override def copy(extra: ParamMap): Uninterceptor[M] = copyValues(new Uninterceptor[M](interceptor.copy(extra)), extra)
  }

  class UninterceptorComposite[M <: LinearModel[M], C <: CombinedModel[M,C]] (val interceptor: Interceptor)
    extends UnwrappedStage.PredefinedDataTransformer[C, UninterceptorComposite[M,C]](Identifiable.randomUID("uninterceptor"), interceptor)
     {

    override def transformModel(model: C, originalData: DataFrame): C = model.transformNested(Interceptor.transformModel)

       override def copy(extra: ParamMap): UninterceptorComposite[M, C] = copyValues(new UninterceptorComposite[M,C](interceptor.copy(extra)), extra)
     }
}
