package odkl.analysis.spark.texts

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.odkl.texts.FreqStatsTransformer
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{ArrayType, LongType, StringType, StructType}
import org.scalatest.FlatSpec

/**
  * Created by eugeny.malyutin on 17.05.16.
  */
class FreqStatsTransformerSpec extends FlatSpec with TestEnv with org.scalatest.Matchers {

  "FreqStatsTransformer" should "count freq" in {
    val fTransformer =  new FreqStatsTransformer()
      .setInputDataCol("data")
      .setOutputColFreq("Freq")
      .setOutputColTerm("Term")

    val schema = new StructType().add("data",ArrayType(StringType,true))
    val inDF = sqlc.createDataFrame(
      sc.parallelize(Seq(Seq[String]("a","b","c"),Seq[String]("a","b","a")))
        .map(f => {Row(f)}), schema)

    val correctAns = Array[(String,Double)](("a",2D/5D),("b",2D/5D),("c",1D/5D))
    val realAns = fTransformer.transform(inDF).sort("Term").collect().map(f =>{(f.getAs[String]("Term"),f.getAs[Double]("Freq"))})
    assertResult(correctAns)(realAns)

  }
  "FreqStatsTransformer" should "filter freq by uni and bi treshold" in {
    val fTransformer =  new FreqStatsTransformer()
      .setInputDataCol("data")
      .setOutputColFreq("Freq")
      .setOutputColTerm("Term")
      .setTresholdArr(Array[Double](1.5D/8D,1.1D/8D))

    val schema = new StructType().add("data",ArrayType(StringType,true))
    val inDF = sqlc.createDataFrame(
      sc.parallelize(Seq(Seq[String]("a","b","c","c a", "c a"),Seq[String]("a","b","a", "c a", "a b")))
        .map(f => {Row(f)}), schema)

    val correctAns = Array[(String,Double)](("a",2D/8D),("b",2D/8D),("c a",2D/8D))
    val realAnsDF = fTransformer.transform(inDF).sort("Term")
      val realAns = realAnsDF.collect().map(f =>{(f.getAs[String]("Term"),f.getAs[Double]("Freq"))})
    assertResult(correctAns)(realAns)

  }

  "FreqStatsTransformer" should "extract max timestamp by term" in {
    val fTransformer =  new FreqStatsTransformer()
      .setInputDataCol("data")
      .setOutputColFreq("Freq")
      .setOutputColTerm("Term")
        .setWithTimestamp(true)
        .setTimestampColumnName("timestamp")
      .setTresholdArr(Array[Double](1D/8D,1.1D/8D))

    val schema =
      new StructType().add("data",ArrayType(StringType,true)).add("timestamp",LongType)
    val inDF = sqlc.createDataFrame(
      sc.parallelize(Seq(Seq(Seq[String]("a","c","c a", "c a"),100L),Seq(Seq[String]("c a", "a b"),150L),Seq(Seq[String]("b"),200L)))
        .map(f => {Row.fromSeq(f)}), schema)

    inDF.collect()
    val correctAns = Array[(String,Double,Long)](("a",1D/6D,100L),("a b",1D/6D, 150L),("b",1D/6D,200L),
      ("c",1D/6D, 100L),("c a",2D/6D, 150L))
    val realAns = fTransformer.transform(inDF).sort("Term").collect().map(f =>{(f.getAs[String]("Term"),f.getAs[Double]("Freq"),f.getAs[Long]("timestamp"))})
    assertResult(correctAns)(realAns)
    assertResult(correctAns(1))(realAns(1))

  }
}