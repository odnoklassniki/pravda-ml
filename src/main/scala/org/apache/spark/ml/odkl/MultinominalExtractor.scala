package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utility for extracting mutli-nominal features (converts string/set of strings
  * to vectors).
  */


import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, BinaryAttribute}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, functions}

/**
  * Parameters for multinominal feature extractor.
  */
trait MultinominalExtractorParams extends HasInputCol with HasOutputCol {
  val values = new StringArrayParam(this, "values", "Predefined set of values.")

  val replacements: Param[Map[String, String]] = JacksonParam.mapParam[String](this, "replacements",
    """Specifies optional replacements for read
      |values. Can be used to merge certain outputs into one.""".stripMargin)

  setDefault(replacements -> Map())

  def setValues(v: String*): this.type = set(values, v.toArray)

  def setInputCol(column: String): this.type = set(inputCol, column)

  def setOutputCol(column: String): this.type = set(outputCol, column)

  def setReplacements(value: Map[String,String]): this.type = set(replacements, value)

  def getValues(): Array[String] = $(values)
}

/**
  * Utility for converting columns with string or a set of stings into a vector of 0/1 with
  * the cardinality equal to the number of unique string values used.
  *
  * @param uid
  */
class MultinominalExtractor(override val uid: String) extends
  Estimator[MultinominalExtractorModel] with MultinominalExtractorParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("multinomialExtractorEstimator"))

  val valuesToIgnore = new StringArrayParam(this, "valuesToIgnore", "Values not to add as classes.")

  setDefault(valuesToIgnore -> Array())

  def setValuesToIgnore(value: String*): this.type = set(valuesToIgnore, value.toArray)

  override def fit(dataset: Dataset[_]): MultinominalExtractorModel = {
    val model: MultinominalExtractorModel = copyValues(
      if (isDefined(values)) {
        new MultinominalExtractorModel().setValues($(values): _*).setReplacements($(replacements))
      } else {

        val collectValues = SparkSqlUtils.reflectionLock.synchronized(
          dataset.schema($(inputCol)).dataType match {
            case string: StringType =>
              functions.udf[Seq[String], String](x => if (x != null) Seq($(replacements).getOrElse(x, x)) else Seq())
            case ArrayType(elementType: StringType, containsNull: Boolean) =>
              functions.udf[Seq[String], Seq[String]](x => if(x != null) x.map(y => $(replacements).getOrElse(y,y)).distinct else Seq())
            case _ => throw new IllegalArgumentException(s"Expected either single string or array of strings at ${$(inputCol)}.")
          })

        val vals = dataset
          .filter(dataset($(inputCol)).isNotNull)
          .select(collectValues(dataset($(inputCol))))
          .rdd
          .flatMap(_.getAs[Seq[String]](0))
          .countByValue().toSeq.filterNot(x => $(valuesToIgnore).contains(x._1)).sortBy(-_._2).map(_._1).toArray

        new MultinominalExtractorModel().setValues(vals: _*).setReplacements($(replacements))
      }.setParent(this), ParamMap())

    logInfo(s"For column ${$(inputCol)} extracted values (${model.getValues().mkString(",")}).")
    model
  }

  override def copy(extra: ParamMap): Estimator[MultinominalExtractorModel] = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    new StructType(schema.fields :+ new StructField($(outputCol), new VectorUDT, true))
}

/**
  * Adds read logic
  */
object MultinominalExtractor extends DefaultParamsReadable[MultinominalExtractor]

/**
  * Model produced by the multinominal extractor. Knows the predefined set of values and
  * maps strings/set of strings to vectors of 0/1 with cardinality equal to amount of known values.
  */
class MultinominalExtractorModel(override val uid: String) extends Model[MultinominalExtractorModel]
  with MultinominalExtractorParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("multinomialExtractor"))

  override def copy(extra: ParamMap): MultinominalExtractorModel = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val indices: Map[String, Int] = $(values).zipWithIndex.toMap

    val createVector = SparkSqlUtils.reflectionLock.synchronized(
      dataset.schema($(inputCol)).dataType match {
        case string: StringType => functions.udf[Vector, String](
          x => if (x == null) {
            Vectors.zeros(indices.size)
          } else {
            indices.get($(replacements).getOrElse(x,x))
              .map(x => Vectors.sparse(indices.size, Array(x), Array(1.0)).compressed)
              .getOrElse(Vectors.zeros(indices.size))
          })
        case ArrayType(elementType: StringType, _) => functions.udf[Vector, Seq[String]](
          x => {
            if (x == null) {
              Vectors.zeros(indices.size)
            } else {
              val ind: Array[Int] = x.map(y => $(replacements).getOrElse(y,y)).filter(indices.contains).map(indices).distinct.sorted.toArray
              Vectors.sparse(indices.size, ind, Array.fill(ind.length)(1.0)).compressed
            }
          }
        )
      })

    val metadata = new AttributeGroup(
      $(inputCol),
      $(values).map(x => BinaryAttribute.defaultAttr.withName(x).asInstanceOf[Attribute]))
      .toMetadata()

    dataset.withColumn(
      $(outputCol),
      createVector(dataset($(inputCol))).as($(outputCol), metadata))
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    new StructType(schema.fields :+ new StructField($(outputCol), new VectorUDT, true))
}

/**
  * Adds read ability
  */
object MultinominalExtractorModel extends DefaultParamsReadable[MultinominalExtractorModel]
