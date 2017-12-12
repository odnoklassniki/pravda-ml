package odkl.analysis.spark.texts

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.odkl.texts.NGramExtractor
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{ArrayType, StringType, StructType}
import org.scalatest.FlatSpec

/**
  * Created by eugeny.malyutin on 17.05.16.
  */
class NGramExtractorSpec extends FlatSpec with TestEnv with org.scalatest.Matchers {

  "NGramExtractor" should "extract NGrams upTo=true" in {
    val nGramExtractor =
      new NGramExtractor()
        .setUpperN(2)
        .setInputCol("textTokenized")
        .setOutputCol("nGram")

    val schema = new StructType().add("textTokenized",ArrayType(StringType,true))
    val inDF = sqlc.createDataFrame(
      sc.parallelize(Seq(Seq[String]("ab","bc","cd"),Seq[String]("a","b")))
        .map(f => {Row(f)}), schema)

    val outDF = nGramExtractor.transform(inDF)

    val outArrays = outDF.collect().map(_.getAs[Seq[String]]("nGram")).toSeq

    val correctArrays = Seq(Seq("ab","bc","cd","ab bc","bc cd"),Seq("a","b", "a b"))
    assertResult(correctArrays)(outArrays)
  }

  "NGramExtractor" should "extract NGrams upTo=false" in {
    val nGramExtractor =
      new NGramExtractor()
        .setUpperN(3)
        .setLowerN(3)
        .setInputCol("textTokenized")
        .setOutputCol("nGram")

    val schema = new StructType().add("textTokenized",ArrayType(StringType,true))
    val inDF = sqlc.createDataFrame(
      sc.parallelize(Seq(Seq[String]("a","b","c","d")).map(f => {Row(f)})),
      schema)

    val outDF = nGramExtractor.transform(inDF)

    val outArrays = outDF.collect().map(_.getAs[Seq[String]]("nGram")).toSeq

    val correctArrays = Seq(Seq("a b c", "b c d"))
    assertResult(correctArrays)(outArrays)
  }
  "NGramExtractor" should "extract NGrams with the same col" in {
    val nGramExtractor =
      new NGramExtractor()
        .setUpperN(3)
        .setLowerN(3)
        .setInputCol("textTokenized")
        .setOutputCol("textTokenized")

    val schema = new StructType().add("textTokenized",ArrayType(StringType,true))
    val inDF = sqlc.createDataFrame(
      sc.parallelize(Seq(Seq[String]("a","b","c","d")).map(f => {Row(f)})),
      schema)

    val outDF = nGramExtractor.transform(inDF)

    val outArrays = outDF.collect().map(_.getAs[Seq[String]]("textTokenized")).toSeq

    val correctArrays = Seq(Seq("a b c", "b c d"))
    assertResult(correctArrays)(outArrays)
  }

}