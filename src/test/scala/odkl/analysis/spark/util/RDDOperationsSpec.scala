package odkl.analysis.spark.util

import odkl.analysis.spark.TestEnv
import org.apache.spark.sql.Row
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by vyacheslav.baranov on 02/12/15.
  */
class RDDOperationsSpec extends FlatSpec with TestEnv with Matchers with SQLOperations {

  import sqlc.implicits._
  import RDDOperations._

  "Group sorted " should " should work with two partitions" in {

    val frame = sc.parallelize(Seq(
      "1" -> "1.1",
      "1" -> "1.2",
      "1" -> "1.3",
      "2" -> "2.1",
      "2" -> "2.2"),
      1)
      .toDF("key", "value")
      .repartition(2, $"key")
      .sortWithinPartitions($"key")

    val data: Array[(String, Array[String])] = frame
      .map(r => r.getString(0) -> r.getString(1))
      .groupWithinPartitionsByKey
      .map(x => x._1 -> x._2.toArray)
      .collect()


    data.length should be(2)

    val first = data.find(_._1 == "1").get._2.sorted
    val second = data.find(_._1 == "2").get._2.sorted

    first should be(Seq("1.1","1.2","1.3"))
    second should be(Seq("2.1","2.2"))
  }


  "Group sorted " should " should work with three partitions" in {

    val frame = sc.parallelize(Seq(
      "1" -> "1.1",
      "1" -> "1.2",
      "1" -> "1.3",
      "2" -> "2.1",
      "2" -> "2.2"),
      1)
      .toDF("key", "value")
      .repartition(3, $"key")
      .sortWithinPartitions($"key")

    val data: Array[(String, Array[Row])] = frame
      .map(x => x)
      .groupWithinPartitionsBy(r => r.getString(0))
      .map(x => x._1 -> x._2.toArray)
      .collect()

    data.length should be(2)

    val first = data.find(_._1 == "1").get._2.map(_.getString(1)).sorted
    val second = data.find(_._1 == "2").get._2.map(_.getString(1)).sorted

    first should be(Seq("1.1","1.2","1.3"))
    second should be(Seq("2.1","2.2"))
  }

  "Group sorted " should " should work with one partitions" in {

    val rdd = sc.parallelize(Seq(
      "1" -> "1.1",
      "1" -> "1.2",
      "1" -> "1.3",
      "2" -> "2.1",
      "2" -> "2.2"),
      1)

    rdd.take(10).foreach(System.out.println)

    val data = rdd.groupWithinPartitionsByKey.map(x => x._1 -> x._2.toArray).collect

    data.length should be(2)

    val first = data.find(_._1 == "1").get._2.sorted
    val second = data.find(_._1 == "2").get._2.sorted

    first should be(Seq("1.1", "1.2", "1.3"))
    second should be(Seq("2.1", "2.2"))
  }

  "Allign " should " allign" in {

    val alligned = IteratorUtils.allignByTime(Iterator(10,12,21,125), (x: Int) => x.toLong, 10)
      .map(x => x._1 -> x._2.toArray).toArray

    alligned.length should be(2)
    alligned(0)._2 should be(Array(10,12,21))
    alligned(1)._2 should be(Array(125))
  }

  "Allign " should " allign 3" in {

    val alligned =  IteratorUtils.allignByTime[Int](Iterator(10,12,21,125), x => x.toLong, 5)
      .map(x => x._1 -> x._2.toArray).toArray

    alligned.length should be(3)
    alligned(0)._2 should be(Array(10,12))
    alligned(1)._2 should be(Array(21))
    alligned(2)._2 should be(Array(125))
  }

}
