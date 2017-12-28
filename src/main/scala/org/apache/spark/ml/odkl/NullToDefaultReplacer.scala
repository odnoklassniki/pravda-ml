package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utilities for replacing nulls in the dataset with some values
  * TODO: Support not only defaults (zeros), but also means.
  */

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{BooleanType, NumericType, StructType}
import org.apache.spark.sql.{DataFrame, functions}

/**
  * Utility used to replace null values with defaults (zero or false).
  */
class NullToDefaultReplacer(override val uid: String) extends Transformer
  with HasColumnsSets with DefaultParamsWritable {

  val defaultValues = JacksonParam.mapParam[String](
    this, "defaultValues", "Default values to assign to columns")

  setDefault(defaultValues, Map[String, String]())

  def this() = this(Identifiable.randomUID("nullToDefaultReplacer"))

  override def transform(dataset: DataFrame): DataFrame = {
    val columns = extractColumns(dataset).map(_.name).toSet

    val expressions = dataset.schema.fields
      .map(field =>
        if (columns.contains(field.name) && field.nullable) {
          val column = field.dataType match {
            case _: BooleanType => functions.coalesce(dataset(field.name), functions.lit($(defaultValues).getOrElse(field.name, "false")).cast(field.dataType))
            case _: NumericType => functions.coalesce(dataset(field.name), functions.lit($(defaultValues).getOrElse(field.name, "0")).cast(field.dataType))
            case _ => dataset(field.name)
          }
          if (field.metadata != null) {
            column.cast(field.dataType).as(field.name, field.metadata)
          } else {
            column.cast(field.dataType).as(field.name)
          }
        } else {
          dataset(field.name)
        })

    dataset.select(expressions: _*)
  }
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema
}

/**
  * Adds read ability.
  */
object NullToDefaultReplacer extends DefaultParamsReadable[NullToDefaultReplacer]
