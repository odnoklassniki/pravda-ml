package org.apache.spark.ml.odkl

import odkl.analysis.spark.TestEnv
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{LongType, StringType, StructType}
import org.scalatest.FlatSpec

/**
  * Created by eugeny.malyutin on 11.09.17.
  */
class TopKTransformerSpec extends FlatSpec with TestEnv with org.scalatest.Matchers{


  "TopKTransformer" should "adequate aggregate and sort with one column to group by" in {

    val data = Seq( ("1", "1", 1L), ("1", "2", 3L), ("1", "1", 2L),
      ("2", "1", 1L), ("2", "1", 3L))
        .map(Row.fromTuple(_))

    val schema =  new StructType().add("op1", StringType)
      .add("op2", StringType)
      .add("value", LongType)

    val testDF =  sqlc.createDataFrame(sqlc.sparkContext.parallelize(data), schema)

    val topKTransformer = new TopKTransformer[Long]()
      .setTopK(2)
      .setColumnToOrderGroupsBy("value")
      .setGroupByColumns("op1")

    val result = topKTransformer.transform(testDF).collect()
        .map(raw => raw.getString(0) -> (raw.getString(1), raw.getLong(2))) //group - subGroup, value
    println(topKTransformer.transform(testDF).collect().map(_.mkString(" _ ")))

    Array(("1",3), ("1",1)) should equal (result.filter(_._1 == "2").map(_._2))
    Array(("2",3), ("1",2)) should equal (result.filter(_._1 == "1").map(_._2))

  }

  "TopKTransformer" should "adequate aggregate and sort with multiple columns to group by" in {

    val data = Seq( ("1", "1", 1L, "interest1"),
      ("1", "1", 3L, "interest2"), //max
      ("1", "1", 2L, "interest3"),
      ("2", "1", 2L, "interest4"), //max
        ("2", "1", 1L, "interest5"))
      .map(Row.fromTuple(_))

    val schema =  new StructType().add("op1", StringType)
      .add("op2", StringType)
      .add("value", LongType)
      .add("data", StringType)

    val testDF =  sqlc.createDataFrame(sqlc.sparkContext.parallelize(data), schema)

    val topKTransformer = new TopKTransformer[Long]()
      .setTopK(1)
      .setColumnToOrderGroupsBy("value")
      .setGroupByColumns("op1", "op2")

    val result = topKTransformer.transform(testDF).collect()
      .map(raw => (raw.getString(0),raw.getString(1)) -> raw.getString(3)) //group, subGroup -> data

    assertResult("interest2") (result.find(_._1 == ("1", "1")).get._2)
    assertResult("interest4") (result.find(_._1 == ("2", "1")).get._2)

  }
}