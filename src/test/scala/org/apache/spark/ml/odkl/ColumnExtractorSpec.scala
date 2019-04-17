package org.apache.spark.ml.odkl

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.odkl.texts.RegexpReplaceTransformer
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StringType, StructType}
import org.scalatest.FlatSpec

class ColumnExtractorSpec extends FlatSpec with TestEnv with org.scalatest.Matchers{

  val data = Seq(
    (1, 5, "hello", "pass"),
    (2, 4, "how", "pass"),
    (3, 3, "are", "pass"),
    (4, 2, "yoy", "pass"),
    (5, 1, "?", "pass"))
    .map(Row.fromTuple(_))

  val schema =  new StructType().add("number1", IntegerType)
    .add("number2", IntegerType)
    .add("text", StringType)
    .add("skip", StringType)

  val testDF =  sqlc.createDataFrame(sqlc.sparkContext.parallelize(data), schema)

  "ColumnExtractor" should "return columns specified in withColumns and withExpresions" in {

    val columnsExtractor = new ColumnsExtractor()
      .withColumns("number1", "number2", "text")
      .withExpresions("sumNumbers" -> "number1 + number2")
      .withExpresions("lengthText" -> "length(text)")

    val result = columnsExtractor.transform(testDF)
    result.columns should be (Array("number1", "number2", "text", "sumNumbers", "lengthText"))

    result.rdd.map(r => (r.getInt(0), r.getInt(1), r.getString(2), r.getInt(3), r.getInt(4))).collect.foreach {
      case (first: Int, second: Int, third: String, fourth: Int, fifth: Int) =>
        fourth should be(first+second)
        fifth should be (third.length())
    }
  }

  "ColumnExtractor" should "return input columns and columns specified in withExpresions" in {

    val columnsExtractor = new ColumnsExtractor()
      .setSaveInputCols(true)
      .withExpresions("sumNumbers" -> "number1 + number2")
      .withExpresions("lengthText" -> "length(text)")

    val result = columnsExtractor.transform(testDF)
    result.columns should be (Array("number1", "number2", "text", "skip", "sumNumbers", "lengthText"))

    result.rdd.map(r => (r.getInt(0), r.getInt(1), r.getString(2), r.getInt(4), r.getInt(5))).collect.foreach {
      case (first: Int, second: Int, third: String, fourth: Int, fifth: Int) =>
        fourth should be(first+second)
        fifth should be (third.length())
    }
  }

  "ColumnExtractor" should "work in pipeline (transformSchema)" in {

    val pipeline = new Pipeline().setStages(Array(
      new ColumnsExtractor()
        .setSaveInputCols(true)
        .withExpresions(
          "length_text" -> "length(text)",
          "upper_p" -> "LENGTH(regexp_replace(text, '[^A-ZА-ЯЁ]', ''))/length(text)",
          "lower_p" -> "LENGTH(regexp_replace(text, '[^a-zа-яё]', ''))/length(text)",
          "digits_p" -> "LENGTH(regexp_replace(text, '[^0-9]', ''))/length(text)"),

      new RegexpReplaceTransformer()
        .setInputCol("text")
        .setOutputCol("text")
        .setRegexpPattern("h")
        .setRegexpReplacement("H")
    )).fit(testDF)

    val result = pipeline.transform(testDF)
    result.columns should be (Array("number1", "number2", "text", "skip", "length_text", "upper_p", "lower_p", "digits_p"))
  }

}
