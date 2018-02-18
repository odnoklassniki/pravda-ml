package org.apache.spark.ml.odkl


import odkl.analysis.spark.util.collection.OpenHashMap
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, functions}

/**
  * Utility used to extract nested values from vectors into dedicated columns. Requires vector metadata and extracts
  * names from where. Typically used as a final stage before results visualization.
  */
class VectorExplode(override val uid: String) extends
  Transformer with DefaultParamsWritable {

  val valueCol = new Param[String](this, "valueCol", "Name of the column to store value name.")

  def setValueCol(value: String) : this.type = set(valueCol, value)

  setDefault(valueCol -> "value")


  def this() = this(Identifiable.randomUID("vectorExplode"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val vectors: Array[StructField] = dataset.schema.fields.filter(_.dataType.isInstanceOf[VectorUDT])

    val resultSchema = StructType(Seq(
      StructField($(valueCol), StringType, nullable = false)) ++
      vectors.map(f => StructField(f.name, DoubleType, nullable = true))
    )

    val arraySize = resultSchema.size - 1

    val names: Array[Map[Int, String]] = vectors.map(
      f => {
        AttributeGroup.fromStructField(f).attributes
          .map(attributes => attributes.filter(_.name.isDefined).map(a => a.index.get -> a.name.get).toMap)
          .getOrElse(Map())
      })

    val maxCapacity = names.map(_.size).max

    val explodeVectors : (Row => Array[Row]) = (r: Row ) => {
      val accumulator = new OpenHashMap[String,Array[Double]](maxCapacity)

      for(i <- 0 until r.length) {
        val vector = r.getAs[Vector](i)

        vector.foreachActive((index, value) => {
          val name = names(i).getOrElse(index, s"${vectors(i).name}_$index")

          accumulator.changeValue(
            name,
            Array.tabulate(arraySize) {ind => if(i == ind) value else Double.NaN},
            v => {v(i) = value; v})
        })
      }

      accumulator.map(x => new GenericRowWithSchema(
        (Seq(x._1) ++ x._2.toSeq.map(v => if (v.isNaN) null else v)).toArray,
        resultSchema)).toArray
    }

    val vectorsStruct = functions.struct(vectors.map(f => dataset(f.name)): _*)
    val explodeUDF = SparkSqlUtils.customUDF(explodeVectors, ArrayType(resultSchema), Some(Seq(vectorsStruct.expr.dataType)))
        
    val expression = functions.explode(explodeUDF(vectorsStruct))

    dataset
      .withColumn(uid, expression)
      .select(
        dataset.schema.fields.filterNot(_.dataType.isInstanceOf[VectorUDT]).map(f => dataset(f.name)) ++
          resultSchema.fields.map(f => functions.expr(s"$uid.${f.name}").as(f.name)) :_*)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    StructType(schema.fields.map(x =>
      x.dataType match {
        case vector: VectorUDT => StructField(x.name, typeFromVector(x))
        case _ => x
      }
    ))

  def typeFromVector(field: StructField): StructType = {
    val attributes = AttributeGroup.fromStructField(field)
    StructType(attributes.attributes
      .map(_.map(a => a.name.getOrElse(s"_${a.index.get}")))
      .getOrElse(Array.tabulate(attributes.size) { i => s"_$i" })
      .map(name => StructField(name, DoubleType, nullable = false)))
  }
}
