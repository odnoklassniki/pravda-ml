package org.apache.spark.ml.odkl

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.sql.{DataFrame, Dataset, functions}
import org.apache.spark.sql.types.{Metadata, StringType, StructField, StructType}

/**
  * Assuming there is a metadata attached to a integer field can be used to replace ints with corresponding
  * attribute names. Used, for example in the validation pipeline to avoid attaching large strings to the validation
  * results (eg. score/label descriptions) before the very end.
  */
class NameAssigner(override val uid: String) extends Transformer with HasInputCols{

  def setInputCols(column: String*) : this.type = set(inputCols, column.toArray)

  def this() = this(Identifiable.randomUID("NameAssigner"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    $(inputCols)

    $(inputCols).foldLeft(dataset.toDF)((data, column) => {
      val metadata: Metadata = dataset.schema(column).metadata
      val attributes = AttributeGroup.fromStructField(
        StructField(column, new VectorUDT, nullable = false, metadata = metadata))

      val map = attributes.attributes
        .map(arr => arr.filter(_.name.isDefined).map(a => a.index.get -> a.name.get).toMap)
        .getOrElse(Map())

      val func = functions.udf[String, Number](x => if(x == null) {
        null
      } else {
        val i = x.intValue()
        map.getOrElse(i, i.toString)
      })

      data.withColumn(column, func(data(column)).as(column, metadata))
    }).toDF
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    StructType(schema.map(f => if ($(inputCols).contains(f.name)) {
      StructField(f.name, StringType, f.nullable, f.metadata)
    } else {
      f
    }))
}


