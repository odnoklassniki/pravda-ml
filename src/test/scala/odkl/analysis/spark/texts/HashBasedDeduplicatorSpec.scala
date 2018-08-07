package odkl.analysis.spark.texts

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.odkl.texts.HashBasedDeduplicator
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.ml.odkl.MatrixUtils
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{LongType, StringType, StructType}
import org.scalatest.FlatSpec

class HashBasedDeduplicatorSpec extends FlatSpec with TestEnv with org.scalatest.Matchers {
  "cotrect HashBasedDeduplicator " should " remove similar vectors based on hash " in {

    val vectorsSize = 10000

    val vector1 = (Vectors.sparse(vectorsSize, Array(5, 6, 7), Array(1.0, 1.0, 1.0)), 1L, "vector1")
    val vector2 = (Vectors.sparse(vectorsSize, Array(5, 6, 7), Array(1.0, 1.0, 0.0)), 1L, "vector2")
    val vector3 = (Vectors.sparse(vectorsSize, Array(5, 6, 7), Array(1.0, 0.0, 1.0)), 2L, "vector3") //pretty similar, but in 2nd bucket
    val vector4 = (Vectors.sparse(vectorsSize, Array(1, 2), Array(1.0, 1.0)), 1L, "vector4") //completly another but in 1-st bucket

    val schema = new StructType()
      .add("vector", MatrixUtils.vectorUDT)
      .add("hash", LongType)
      .add("alias", StringType)

    val dataFrame = sqlc.createDataFrame(sc.parallelize(Seq(vector1, vector2, vector3, vector4).map(Row.fromTuple(_))), schema)
    val deduplicator = new HashBasedDeduplicator()
      .setInputColHash("hash")
      .setInputColVector("vector")
      .setSimilarityTreshold(0.80)

   val answer = deduplicator.transform(dataFrame)
        .collect().map(row => (row.getLong(1), row.getString(2)))

    assert(answer.exists(_._2 == "vector1")) //should stay
    assert(!answer.exists(_._2 == "vector2")) //should be removed
    assert(answer.exists(_._2 == "vector3")) //should stay cause in other bucket (FalseNegative)
    assert(answer.exists(_._2 == "vector4")) //should stay cause different (FalsePositive)
  }
}
