package odkl.analysis.spark.texts

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.odkl.texts.RandomProjectionsHasher
import org.apache.spark.mllib.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.scalatest.FlatSpec

class RandomProjectionsHasherSpec extends FlatSpec with TestEnv with org.scalatest.Matchers {

  case class VectorWrapper(data: Vector)

  "RPH" should "put the same vectors into same buckets" in {

    val vectorsSize = 10000
    val vector1 = Vectors.sparse(vectorsSize, Array(5, 6, 7), Array(1.0, 1.0, 1.0))

    val vector2 = Vectors.sparse(vectorsSize, Array(5, 6, 7), Array(1.0, 1.0, 0.0))

    val vectorsSchema = new StructType()
      .add("data", new VectorUDT)

    val data = sqlc.createDataFrame(
      sc.parallelize(Seq(Seq(vector1), Seq(vector2)))
        .map(Row.fromSeq(_)), vectorsSchema)


    val hasher = new RandomProjectionsHasher()
      .setBasisSize(4)
      .setDim(vectorsSize)
      .setSparsity(0.5)
      .setInputCol("data")
      .setOutputCol("hash")

    val hashed = hasher.transform(data).collect().map(_.getLong(1))

    assert(hashed(0) == hashed(1))

  }
}