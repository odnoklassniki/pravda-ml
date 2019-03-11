package org.apache.spark.ml.odkl.texts

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamPair, ParamValidators, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{ArrayType, StringType, StructType}

/**
  * Created by eugeny.malyutin on 17.05.16.
  * Simple NGramExtractor Transformer with option to extract from lowerNGrams to UPper together
  */
class NGramExtractor(override val uid: String)
  extends Transformer
    with DefaultParamsWritable
    with Params
    with HasInputCol
    with HasOutputCol {

  val upperN: IntParam = new IntParam(this, "LowerN", "number elements(lower) per n-gram (>=1) lowerN<=n<=upperN",
    ParamValidators.gtEq(1))

  val lowerN: IntParam = new IntParam(this, "UpperN", "number elements per n-gram (>=1)",
    ParamValidators.gtEq(1))

  def this() = this(Identifiable.randomUID("nGramExtractor"))

  /** @group setParam */
  def setUpperN(value: Int): this.type = set(upperN, value)

  /** @group getParam */
  def getUpperN: Int = $(upperN)

  /** @group setParam */
  def setLowerN(value: Int): this.type = set(lowerN, value)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(new ParamPair[Int](upperN, 2), new ParamPair[Int](lowerN, 1))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val lowerBound = $(lowerN)
    val upperBound = $(upperN)
    val nGramUDF = udf[Seq[String], Seq[String]](NGramUtils.nGramFun(_,lowerBound,upperBound))
    dataset.withColumn($(outputCol), nGramUDF(dataset.col($(inputCol))))
  }


  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    if ($(inputCol) != $(outputCol)) {
      schema.add($(outputCol), new ArrayType(StringType, true))
    } else {
      schema
    }
  }
}
object NGramExtractor extends DefaultParamsReadable[NGramExtractor] {
  override def load(path: String): NGramExtractor = super.load(path)
}


