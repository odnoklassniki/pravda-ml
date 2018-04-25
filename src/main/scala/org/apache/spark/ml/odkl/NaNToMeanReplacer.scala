package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utility used to handle unknown values (NaN's in feature vectors).
  * Replaces unknowns with mean values before proceeding to training.
  */

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.odkl.NaNToMeanReplacerModel.NaNSafeVectorMean
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, DefaultParamsReader, Identifiable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.{DataFrame, Row, functions}

/**
  * Set of parameters for the replacer
  */
trait NaNToMeanReplacerParams extends HasInputCol with HasOutputCol {

  final val groupByColumn = new Param[String](
    this, "groupByColumn", "Grouping criteria for the evaluation. Means are computed in the scope of the group" +
      " to support better granularity.")

  def setGroupByColumn(column: String): this.type = set(groupByColumn, column)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)
}

/**
  * Estimates mean values ignoring NaN's
  */
class NaNToMeanReplacerEstimator(override val uid: String) extends Estimator[NaNToMeanReplacerModel]
  with NaNToMeanReplacerParams {
  def this() = this(Identifiable.randomUID("nanToMeanReplacerEstimator"))

  override def fit(dataset: DataFrame): NaNToMeanReplacerModel = {

    val size = AttributeGroup.fromStructField(dataset.schema($(inputCol))).size
    val mean = new NaNSafeVectorMean(size)

    // TODO: Should we support median to?
    val means = dataset
      .groupBy($(groupByColumn))
      .agg(mean(dataset($(inputCol))))
      .map(r => {
        r.getString(0) -> r.getAs[Vector](1)
      })
      .collect()
      .toMap


    new NaNToMeanReplacerModel()
      .setInputCol($(inputCol))
      .setOutputCol($(outputCol))
      .setGroupByColumn($(groupByColumn))
      .setMeans(means)
      .setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[NaNToMeanReplacerModel] = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema
}

/**
  * Model used to replace values with pre-computed defaults before training/predicting.
  */
class NaNToMeanReplacerModel(override val uid: String) extends Model[NaNToMeanReplacerModel]
  with NaNToMeanReplacerParams with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("nanToMeanReplacer"))

  val defaults = new Param[Map[String, Vector]](
    this, "defaults", "Vector with the default values for replace.") {
    override def jsonEncode(value: Map[String, Vector]): String = {
      val values: Map[String, String] = value.mapValues(_.toJson)
      JacksonParam.objectMapper.writeValueAsString(values)
    }

    override def jsonDecode(json: String): Map[String, Vector] = {
      val raw = JacksonParam.objectMapper.readValue[Map[String, String]](json, classOf[Map[String, String]])
      raw.transform((key, value) => Vectors.fromJson(value))
    }
  }

  def setMeans(value: Map[String, Vector]): this.type = set(defaults, value)

  override def copy(extra: ParamMap): NaNToMeanReplacerModel = defaultCopy(extra)

  override def transform(dataset: DataFrame): DataFrame = {
    val localDefaults = $(defaults)

    val replacer = functions.udf[Vector, String, Vector]((group, vector) => {
      val defaults = localDefaults(group)
      val copied = vector.copy

      copied match {
        case dense: DenseVector =>
          val values = dense.values
          for (i <- values.indices) {
            if (values(i).isNaN) {
              values(i) = defaults(i)
            }
          }
          dense
        case sparse: SparseVector =>
          val values = sparse.values
          for (i <- values.indices) {
            if (values(i).isNaN) {
              values(i) = defaults(sparse.indices(i))
            }
          }
          sparse
      }
    })

    dataset.withColumn(
      $(outputCol),
      replacer(dataset($(groupByColumn)), dataset($(inputCol))),
      dataset.schema($(inputCol)).metadata)
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema
}

/**
  * Adds support for reading.
  */
object NaNToMeanReplacerModel extends DefaultParamsReadable[NaNToMeanReplacerModel] {

  /**
    * Supplementary UDAF for NaN-safe mean calculation.
    */
  class NaNSafeVectorMean(val dimension: Int) extends UserDefinedAggregateFunction {
    override def inputSchema: StructType = new StructType().add("vector", new VectorUDT())

    override def deterministic: Boolean = true

    override def bufferSchema: StructType = new StructType().add("sum", new VectorUDT()).add("count", new VectorUDT())

    override def dataType: DataType = new VectorUDT

    override def initialize(buffer: MutableAggregationBuffer): Unit = {
      buffer.update(0, Vectors.zeros(dimension))
      buffer.update(1, Vectors.zeros(dimension))
    }

    override def update(buffer: MutableAggregationBuffer, input: Row): Unit = if (input != null) {
      val sum: DenseVector = buffer.getAs[Vector](0).toDense
      val count: DenseVector = buffer.getAs[Vector](1).toDense
      val vector = input.getAs[Vector](0)

      require(vector.size == sum.size, s"Expected vector of dimension ${sum.size}")
      for (i <- 0 until sum.size) {
        if (!vector(i).isNaN) {
          sum.values(i) += vector(i)
          count.values(i) += 1
        }
      }

      buffer.update(0, sum)
      buffer.update(1, count)
    }

    override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
      val sum: DenseVector = buffer1.getAs[Vector](0).toDense
      buffer2.getAs[Vector](0).foreachActive((i, v) => {
        sum.values(i) += v
      })

      val count: DenseVector = buffer1.getAs[Vector](1).toDense
      buffer2.getAs[Vector](1).foreachActive((i, v) => {
        count.values(i) += v
      })

      buffer1.update(0, sum)
      buffer1.update(1, count)
    }

    override def evaluate(buffer: Row): Any = {
      val sum: DenseVector = buffer.getAs[Vector](0).toDense
      val count: DenseVector = buffer.getAs[Vector](1).toDense

      count.foreachActive((i, v) => if (v != 0) sum.values(i) /= v)

      sum
    }
  }

}
