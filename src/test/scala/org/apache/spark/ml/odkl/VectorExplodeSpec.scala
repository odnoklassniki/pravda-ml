package org.apache.spark.ml.odkl

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.{functions, Row}
import org.apache.spark.sql.types.{StructType, StructField, DoubleType}
import org.scalatest.FlatSpec

/**
  * Created by dmitriybugaichenko on 25.01.16.
  */
class VectorExplodeSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations with WithModels with HasMetricsBlock {

  case class Point(id: Int, vector: Vector, mean: Vector)

  lazy val data = sqlc.createDataFrame(Seq(
    Point(1, Vectors.dense(1.0, 3.0), Vectors.dense(10.0, 30.0)),
    Point(2, Vectors.dense(2.0, 4.0), Vectors.sparse(2, Array(1), Array(20.0)))
  ))

  lazy val withMetadata = data.withColumn(
    "vector",
    data("vector").as("vector", new AttributeGroup("vector", Array[Attribute](
      NumericAttribute.defaultAttr.withName("fixed"),
      NumericAttribute.defaultAttr.withName("var")
    )).toMetadata()))
    .withColumn(
      "mean",
      data("mean").as("mean", new AttributeGroup("vector", Array[Attribute](
        NumericAttribute.defaultAttr.withName("fixed"),
        NumericAttribute.defaultAttr.withName("var")
      )).toMetadata()))

  lazy val explode = new VectorExplode().transform(withMetadata)

  "Explode " should " add data" in {
    val result = explode.orderBy("id", "value").collect()

    result(0).getInt(0) should be(1)
    result(0).getString(1) should be("fixed")
    result(0).getDouble(2) should be(1.0)
    result(0).getDouble(3) should be(10.0)

    result(1).getInt(0) should be(1)
    result(1).getString(1) should be("var")
    result(1).getDouble(2) should be(3.0)
    result(1).getDouble(3) should be(30.0)

    result(2).getInt(0) should be(2)
    result(2).getString(1) should be("fixed")
    result(2).getDouble(2) should be(2.0)
    result(2).isNullAt(3) should be(true)

    result(3).getInt(0) should be(2)
    result(3).getString(1) should be("var")
    result(3).getDouble(2) should be(4.0)
    result(3).getDouble(3) should be(20.0)
  }

  "Explode " should " create schema" in {
    val fields = explode.schema.fields

    fields(0).name should be("id")
    fields(1).name should be("value")
    fields(2).name should be("vector")
    fields(3).name should be("mean")
  }
}
