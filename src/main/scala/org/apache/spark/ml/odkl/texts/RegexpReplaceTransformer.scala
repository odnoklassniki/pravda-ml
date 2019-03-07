package org.apache.spark.ml.odkl.texts

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, StringType, StructType}

/**
  * Created by eugeny.malyutin on 12.05.16.
  *
  * regexp_replace in Transformer
  */
class RegexpReplaceTransformer(override val uid: String) extends Transformer
  with HasInputCol
  with HasOutputCol
  with Params
  with DefaultParamsWritable{

  val regexpPattern = new Param[String](this,"regexpPattern","pattern in regexp_replace")

  val regexpReplacement = new Param[String](this,"regexpReplacement","replacement in sql.regexp_replace")

  setDefault(new ParamPair[String](regexpReplacement," "),
    new ParamPair[String](regexpPattern,"[\\.\\,\\\\/#!$0-9<\\*>\\?'\"\\-@%\\\\^&\\\\*;:{}=\\\\-_`~()]"))

  /** @group setParam */
  def setRegexpPattern(value: String): this.type = set(regexpPattern, value)

  /** @group setParam */
  def setRegexpReplacement(value: String): this.type = set(regexpReplacement, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  def this() = this(Identifiable.randomUID("RegexpReplaceTransformer"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.withColumn($(outputCol), regexp_replace(dataset.col($(inputCol)), $(regexpPattern), $(regexpReplacement)))
  }
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    if ($(inputCol) equals $(outputCol)) {
      val schemaWithoutInput = new StructType(schema.fields.filterNot(_.name equals $(inputCol)))
      SchemaUtils.appendColumn(schemaWithoutInput, $(outputCol), ArrayType(StringType, true))
    } else {
      SchemaUtils.appendColumn(schema, $(outputCol), ArrayType(StringType, true))
    }
  }

}

object RegexpReplaceTransformer extends DefaultParamsReadable[RegexpReplaceTransformer] {
  override def load(path: String): RegexpReplaceTransformer = super.load(path)
}
