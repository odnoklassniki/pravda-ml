package org.apache.spark.ml.odkl

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.DenseVector
import org.scalatest.{FlatSpec, FunSuite}

class NullToDefaultReplacerSpec  extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations {
  case class P(id: Int, vector: Vector)

    lazy val data = sqlc.createDataFrame(Seq(
      P(1, null),
      P(2, Vectors.dense(0.0, 1.0)),
      P(3, Vectors.dense(0.0, 2.0))
    ))

  lazy val replacer = new NullToDefaultReplacer()

  lazy val dataWithoutNull = replacer.transform(data).rdd.map(row => (row.getInt(0), row.getAs[DenseVector](1))).collect().sortBy(_._1)

  "Replacer " should " replace null with column type vector to vector zeros" in {

    dataWithoutNull(0)._2 should be(Vectors.dense(0.0, 0.0))
  }


}
