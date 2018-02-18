package org.apache.spark.ml.odkl

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/**
  * Simple utility used to apply SQL WHERE filter
  */
class SqlFilter(override val uid: String) extends
  Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("sqlFilter"))

  val where = new Param[String](this, "where", "WHERE clause to filter the relation")

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.where($(where)).toDF
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema

  def setWhere(value: String) : this.type = set(where, value)
}

object SqlFilter extends DefaultParamsReadable[SqlFilter]