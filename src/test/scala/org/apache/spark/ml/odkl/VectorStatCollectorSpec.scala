package org.apache.spark.ml.odkl

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.ml.attribute.{Attribute, NumericAttribute, AttributeGroup}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import org.scalatest.FlatSpec

/**
  * Created by dmitriybugaichenko on 25.01.16.
  */
class VectorStatCollectorSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations with WithModels with HasMetricsBlock {

  case class Point(id: Int, vector: Vector)

  lazy val data = sqlc.createDataFrame(Seq(
    Point(1, Vectors.dense(1.0, 1.0)),
    Point(1, Vectors.dense(1.0, 2.0)),
    Point(1, Vectors.dense(1.0, 3.0)),
    Point(1, Vectors.dense(1.0, 4.0)),
    Point(1, Vectors.dense(1.0, 5.0)),
    Point(1, Vectors.dense(1.0, 6.0)),
    Point(1, Vectors.dense(1.0, 7.0)),
    Point(1, Vectors.dense(1.0, 8.0)),
    Point(1, Vectors.dense(1.0, 9.0)),
    Point(1, Vectors.dense(1.0, 10.0)),
    Point(2, Vectors.dense(2.0, 1.0)),
    Point(2, Vectors.dense(2.0, 1.0)),
    Point(2, Vectors.dense(2.0, 1.0)),
    Point(2, Vectors.dense(2.0, 2.0)),
    Point(2, Vectors.dense(2.0, 2.0))
  ))

  lazy val statFrame = new VectorStatCollector()
    .setGroupByColumns("id")
    .setInputCol("vector")
    .setDimensions(2)
    .transform(data)

  lazy val stat: Map[Int, Row] =
    statFrame.rdd.map(r => r.getInt(0) -> r).collect().toMap

  lazy val withMetadata = new VectorStatCollector()
    .setGroupByColumns("id")
    .setInputCol("vector")
    .transform(data.withColumn("vector", data("vector").as("vector", new AttributeGroup("vector", Array[Attribute](
      NumericAttribute.defaultAttr.withName("fixed"),
      NumericAttribute.defaultAttr.withName("var")
    )).toMetadata())))

  lazy val noGroups = new VectorStatCollector()
    .setInputCol("vector")
    .transform(data.withColumn("vector", data("vector").as("vector", new AttributeGroup("vector", Array[Attribute](
      NumericAttribute.defaultAttr.withName("fixed"),
      NumericAttribute.defaultAttr.withName("var")
    )).toMetadata())))
    .collect()

  "Collector " should " calculate count" in {

    val field = statFrame.schema.fieldIndex("vector_count")
    stat(1).getLong(field) should be(10l)
    stat(2).getLong(field) should be(5l)
  }

  "Collector " should " calculate mean" in {

    val field = statFrame.schema.fieldIndex("vector_mean")
    stat(1).getAs[Vector](field)(0) should be(1.0)
    stat(1).getAs[Vector](field)(1) should be(5.5)
    stat(2).getAs[Vector](field)(0) should be(2.0)
    stat(2).getAs[Vector](field)(1) should be(7.0 / 5)
  }

  "Collector " should " calculate min" in {

    val field = statFrame.schema.fieldIndex("vector_min")
    stat(1).getAs[Vector](field)(0) should be(1.0)
    stat(1).getAs[Vector](field)(1) should be(1.0)
    stat(2).getAs[Vector](field)(0) should be(2.0)
    stat(2).getAs[Vector](field)(1) should be(1.0)
  }

  "Collector " should " calculate max" in {

    val field = statFrame.schema.fieldIndex("vector_max")
    stat(1).getAs[Vector](field)(0) should be(1.0)
    stat(1).getAs[Vector](field)(1) should be(10.0)
    stat(2).getAs[Vector](field)(0) should be(2.0)
    stat(2).getAs[Vector](field)(1) should be(2.0)
  }

  "Collector " should " calculate variance" in {

    val field = statFrame.schema.fieldIndex("vector_var")
    stat(1).getAs[Vector](field)(0) should be(0.0)
    stat(1).getAs[Vector](field)(1) should be(9 + 1.0 / 6)
    stat(2).getAs[Vector](field)(0) should be(0.0)
    stat(2).getAs[Vector](field)(1) should be(0.3)
  }

  "Collector " should " calculate p10" in {

    val field = statFrame.schema.fieldIndex("vector_p10")
    stat(1).getAs[Vector](field)(0) should be(1.0)
    stat(1).getAs[Vector](field)(1) should be(1.5 +- 0.1)
    stat(2).getAs[Vector](field)(0) should be(2.0)
    stat(2).getAs[Vector](field)(1) should be(1.0 +- 0.1)
  }

  "Collector " should " calculate p50" in {

    val field = statFrame.schema.fieldIndex("vector_p50")
    stat(1).getAs[Vector](field)(0) should be(1.0)
    stat(1).getAs[Vector](field)(1) should be(5.5 +- 0.1)
    stat(2).getAs[Vector](field)(0) should be(2.0)
    stat(2).getAs[Vector](field)(1) should be(1.0 +- 0.1)
  }

  "Collector " should " calculate p90" in {

    val field = statFrame.schema.fieldIndex("vector_p90")
    stat(1).getAs[Vector](field)(0) should be(1.0)
    stat(1).getAs[Vector](field)(1) should be(9.5 +- 0.1)
    stat(2).getAs[Vector](field)(0) should be(2.0)
    stat(2).getAs[Vector](field)(1) should be(2.0 +- 0.1)
  }

  "Collector " should " calculate L1 norm" in {

    val field = statFrame.schema.fieldIndex("vector_L1")
    stat(1).getAs[Vector](field)(0) should be(10.0)
    stat(1).getAs[Vector](field)(1) should be(55)
    stat(2).getAs[Vector](field)(0) should be(10.0)
    stat(2).getAs[Vector](field)(1) should be(7.0)
  }

  "Collector " should " calculate L2 norm" in {

    val field = statFrame.schema.fieldIndex("vector_L2")
    stat(1).getAs[Vector](field)(0) should be(Math.sqrt(10.0))
    stat(1).getAs[Vector](field)(1) should be(Math.sqrt(385))
    stat(2).getAs[Vector](field)(0) should be(Math.sqrt(20))
    stat(2).getAs[Vector](field)(1) should be(Math.sqrt(11))
  }

  "Collector " should " add metadata" in {

    withMetadata.schema.fields.drop(2).foreach(f => {
      val attributes = AttributeGroup.fromStructField(f)

      attributes.size should be (2)
      attributes(0).name.get should be("fixed")
      attributes(1).name.get should be("var")
    })
  }

  "Collector " should " be able to aggregate all" in {
    noGroups.size should be(1)
    noGroups(0).getLong(0) should be(15)
    noGroups(0).getAs[Vector](1)(0) should be(20.0 / 15.0)
    noGroups(0).getAs[Vector](1)(1) should be(62.0 / 15.0 +- 1e-10)
  }
}
