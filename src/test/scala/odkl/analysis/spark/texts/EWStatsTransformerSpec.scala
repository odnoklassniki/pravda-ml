package odkl.analysis.spark.texts

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.odkl.texts.EWStatsTransformer
import org.apache.spark.ml.odkl.texts.EWStatsTransformer.EWStruct
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.scalatest.FlatSpec

/**
  * Created by eugeny.malyutin on 17.05.16.
  */


class EWStatsTransformerSpec extends FlatSpec with TestEnv with org.scalatest.Matchers {

  import sqlc.implicits._

  case class dummyCase(Term: String, sig: Double, ewma: Double, ewmvar: Double)

  case class ewStruct(sig: Double, ewma: Double, ewmvar: Double) extends Serializable

  "CorrectEWFreqStatsTransformer" should "count existing and non-existing today words" in {

    val oldData = Seq(Seq("a", 0.0, 0.1, 0.01), Seq("b", 0.0, 0.2, 0.02), Seq("c", 0.0, 0.3, 0.015))

    val oldDF =
      sqlc.createDataFrame(sc.parallelize(oldData).map(f => {
        Row.fromSeq(f)
      }), new StructType().add("term", StringType)
        .add("sig", DoubleType).add("ewma", DoubleType).add("ewmvar", DoubleType))
    val rddRes = oldDF.
      map { case Row(term, sig, ewma, ewmvar) => Row(term, Row(sig, ewma, ewmvar)) }

    val schemaRes = StructType(
      StructField("term", StringType, false) ::
        StructField("ewStruct", StructType(
          StructField("sig", DoubleType, false) ::
          StructField("ewma", DoubleType, false) ::
          StructField("ewmvar", DoubleType, false) :: Nil
        ), true) :: Nil
    )
    val modernOldDF = sqlc.createDataFrame(rddRes, schemaRes)
      .withColumnRenamed("ewStruct", "old_EWStruct").withColumnRenamed("term", "old_Term")

    oldDF.collect()
    val fTransformer =
      new EWStatsTransformer()
        .setAlpha(0.7)
        .setBeta(0.055)
        .setInputFreqColName("Freq")
        .setInputTermColName("Term")
        .setOldEWStructColName("old_EWStruct")
        .setNewEWStructColName("EWStruct")
        .setOldTermColName("old_Term")
    val schema = new StructType().add("Term", StringType).add("Freq", DoubleType)

    val inDF = sqlc.createDataFrame(
      sc.parallelize(Seq(("a", 0.2), ("b", 0.1), ("d", 0.05)))
        .map(f => {
          Row.fromSeq(Seq(f._1, f._2))
        }), schema)
    val joined = inDF.join(modernOldDF, $"Term" === $"old_Term", "outer")
    val outDF = fTransformer.transform(joined)
    val ans: Array[Row] = outDF.sort("Term").collect()
    assertResult(4)(ans.size)
  }

  "CorrectEWStatsTransformer" should "count EWStats correct" in {

    val mathTransformFun: (String, Double, Double, Double) => EWStruct = EWStatsTransformer.termEWStatsComputing(_:String,_:Double,_:Double,_:Double,0.7,0.005)
    val input = ("test", 0.01, 0.006, 0.003)
    val expected = (0.0669, 0.0088, 0.0009)
    val real = mathTransformFun(input._1, input._2, input._3, input._4)
    val realRounded = (Math.round(real.sig * 10000D) / 10000D, Math.round(real.ewma * 10000D) / 10000D, Math.round(real.ewmvar * 10000D) / 10000D)
    assertResult(expected)(realRounded)
  }
}