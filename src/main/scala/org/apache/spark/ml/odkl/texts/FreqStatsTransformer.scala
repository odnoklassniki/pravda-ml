package org.apache.spark.ml.odkl.texts

/**
  * Created by eugeny.malyutin on 23.05.16.
  */

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.HasInputCol
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructType}
import org.apache.spark.sql.{Column, DataFrame, Dataset}

/**
  * Created by eugeny.malyutin on 06.05.16.
  *
  * Transformer to count Term - Freq for text corpus distinct per document ([T1 T1 T2],[T2] -> {T1 -> 1/3 ,T2 -> 2/3})
  */
class FreqStatsTransformer(override val uid: String) extends Transformer with Params with HasInputCol {

  val distinct = udf((text: Seq[String]) => {
    (text).distinct
  })
  val corpusLen = new Param[Long](this, "CorpusLength", "corpus length if not set - count corpus length by $(outputColTerm)", ParamValidators.gt(0.0))
  val outputColTerm = new Param[String](this, "outputColTerm",
    "output Col with Term[String] name")
  val outputColFreq = new Param[String](this, "outputColFreq",
    "output Col with Freq[Double] name")
  val freqTreshArr = new DoubleArrayParam(this, "FreqTresholdArray", "Array(UniTreshold,BiGramTreshold,TriGramTreshold and etc)")
  val delimeter = new Param[String](this, "Delimiter", "Delimiter for nGrams")
  val withTimestamp = new BooleanParam(this, "WithTimestamp", "should it create max(timestamp) column?")
  val timetstampColumnName = new Param[String](this, "TimestampColumnName", "column with message timestamp and name for column with term's last seen timestamp")

  def this() = this(Identifiable.randomUID("freqStatsTransformer"))

  override def copy(extra: ParamMap): Transformer = {
    defaultCopy(extra)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    dayToTermStat(dataset.toDF)
  }

  def dayToTermStat(preprocedDF: DataFrame): DataFrame = {

    import org.apache.spark.sql.functions._
    val nGramDF: DataFrame =
      if ($(withTimestamp)) {
        preprocedDF.select(explode(distinct(preprocedDF($(inputCol)))).as($(outputColTerm)),preprocedDF($(timetstampColumnName)))
      } else {
        preprocedDF.select(explode(distinct(preprocedDF($(inputCol)))).as($(outputColTerm)))
      }

    val corpusLength = if (isSet(corpusLen)) $(corpusLen) else nGramDF.count()
    val termCountToRelFreq = udf((count: Long) => {
      count.toDouble / corpusLength.toDouble
    })
    val termCountColName = "termCount" + uid

    {
      if (!$(withTimestamp)) {
        nGramDF
          .groupBy($(outputColTerm))
          .agg(count($(outputColTerm)).alias(termCountColName))
      }
      else {
        nGramDF
          .groupBy($(outputColTerm))
          .agg(count($(outputColTerm)).alias(termCountColName), max($(timetstampColumnName)) as $(timetstampColumnName))
      }
    }
      .withColumn($(outputColFreq), termCountToRelFreq(col(termCountColName)))
      .where(isValid(col($(outputColTerm)), col($(outputColFreq))))
      .drop(termCountColName)
  }

  def isValid(term: Column, freq: Column): Column = {
    val arrTresholds = $(freqTreshArr)
    val delim = $(delimeter)
    val udfCreateFreqTreshForTerm = udf((term: String) => {
      val nGramN = term.r.findAllIn(delim).length
      arrTresholds(
        if (nGramN > (arrTresholds.length - 1)) (arrTresholds.length - 1) else nGramN) //if nGramN more then given - use maximum treshold
    })

    val freqTresh = udfCreateFreqTreshForTerm(term)
    (freq > freqTresh)
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    val answerStruct = new StructType()
      .add($(outputColTerm), StringType)
      .add($(outputColFreq), DoubleType)

    if ($(withTimestamp)) answerStruct.add($(timetstampColumnName),LongType) else answerStruct
  }


  /** @group getParam */
  def getInputDataCol: String = $(inputCol)

  /** @group setParam */
  def setInputDataCol(value: String): this.type = set(inputCol, value)

  /** @group getParam */
  def getOutputColTerm: String = $(outputColTerm)

  /** @group setParam */
  def setTresholdArr(value: Array[Double]): this.type = set(freqTreshArr, value)

  /** @group setParam */
  def setDelimiter(value: String): this.type = set(delimeter, value)


  /** @group setParam */
  def setOutputColTerm(value: String): this.type = set(outputColTerm, value)

  /** @group getParam */
  def getOutputColFreq: String = $(outputColFreq)

  /** @group setParam */
  def setOutputColFreq(value: String): this.type = set(outputColFreq, value)

  /** @group setParam */
  def setCorpusLength(value: Long): this.type = set(corpusLen, value)

  /** @group setParam */
  def setWithTimestamp(value: Boolean): this.type = set(withTimestamp, value)

  /** @group setParam */
  def setTimestampColumnName(value: String): this.type = set(timetstampColumnName, value)

  setDefault(

    new ParamPair[Array[Double]](freqTreshArr, Array(1E-8, 1E-6)),
    new ParamPair[String](delimeter, " "),
    new ParamPair[Boolean](withTimestamp,false),
    new ParamPair[String](timetstampColumnName,"timestamp")
  )
}