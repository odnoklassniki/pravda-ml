package org.apache.spark.ml.odkl

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.ml.odkl.UnwrappedStage.{OrderedCutEstimator, OrderedCut}
import org.apache.spark.sql.functions
import org.scalatest.FlatSpec

/**
  * Created by dmitriybugaichenko on 25.01.16.
  */
class OrderedCutSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations with WithModels with HasMetricsBlock {
  lazy val data = OrderedCutSpec._data
  lazy val transformed = OrderedCutSpec._transformed
  lazy val transformedAscending = OrderedCutSpec._transformedAscending
  lazy val transformedMultigroup = OrderedCutSpec._transformedMultiGroup
  val dateToInt = functions.udf[Int,String](x => x.replace("-", "").toInt)

  "Cut " should " should preserve at least as much records as desired" in {
    val perTypeCounts = transformed.groupBy("key").count().collect().map(x => x.getString(0) -> x.getLong(1)).toMap

    perTypeCounts("small") should be(90L)
    perTypeCounts("medium") should be(300L)
    perTypeCounts("large") should be(360L)
  }

  "Cut " should " should select proper dates" in {
    val perTypeDates = transformed
      .withColumn("date",dateToInt(transformed("date")))
      .groupBy("key")
      .max("date")
      .collect()
      .map(x => x.getString(0) -> x.getInt(1)).toMap

    perTypeDates("small") should be(20150202)
    perTypeDates("medium") should be(20150202)
    perTypeDates("large") should be(20150202)
  }

  "Cut " should " should select proper order" in {
    val perTypeDates = transformed
      .withColumn("date",dateToInt(transformed("date")))
      .groupBy("key")
      .min("date")
      .collect()
      .map(x => x.getString(0) -> x.getInt(1)).toMap

    perTypeDates("small") should be(20150127)
    perTypeDates("medium") should be(20150127)
    perTypeDates("large") should be(20150130)
  }

  "Cut " should " should preserve at least as much records as desired (ascending)" in {
    val perTypeCounts = transformedAscending.groupBy("key").count().collect().map(x => x.getString(0) -> x.getLong(1)).toMap

    perTypeCounts("small") should be(90L)
    perTypeCounts("medium") should be(300L)
    perTypeCounts("large") should be(360L)
  }

  "Cut " should " should select proper dates (ascending)" in {
    val perTypeDates = transformedAscending
      .withColumn("date",dateToInt(transformed("date")))
      .groupBy("key")
      .max("date")
      .collect()
      .map(x => x.getString(0) -> x.getInt(1)).toMap

    perTypeDates("small") should be(20150202)
    perTypeDates("medium") should be(20150202)
    perTypeDates("large") should be(20150129)
  }

  "Cut " should " should select proper order (ascending)" in {
    val perTypeDates = transformedAscending
      .withColumn("date",dateToInt(transformed("date")))
      .groupBy("key")
      .min("date")
      .collect()
      .map(x => x.getString(0) -> x.getInt(1)).toMap

    perTypeDates("small") should be(20150127)
    perTypeDates("medium") should be(20150127)
    perTypeDates("large") should be(20150127)
  }

  "Cut " should " should preserve at least as much records as desired (multigroup)" in {
    val perTypeCounts = transformedMultigroup.groupBy("key").count().collect().map(x => x.getString(0) -> x.getLong(1)).toMap

    perTypeCounts("small") should be(90L)
    perTypeCounts("medium") should be(300L)
    perTypeCounts("large") should be(360L)
  }

  "Cut " should " should select proper dates (multigroup)" in {
    val perTypeDates = transformedMultigroup
      .withColumn("date",dateToInt(transformed("date")))
      .groupBy("key")
      .max("date")
      .collect()
      .map(x => x.getString(0) -> x.getInt(1)).toMap

    perTypeDates("small") should be(20150202)
    perTypeDates("medium") should be(20150202)
    perTypeDates("large") should be(20150129)
  }

  "Cut " should " should select proper order (multigroup)" in {
    val perTypeDates = transformedMultigroup
      .withColumn("date",dateToInt(transformed("date")))
      .groupBy("key")
      .min("date")
      .collect()
      .map(x => x.getString(0) -> x.getInt(1)).toMap

    perTypeDates("small") should be(20150127)
    perTypeDates("medium") should be(20150127)
    perTypeDates("large") should be(20150127)
  }
}

object OrderedCutSpec extends TestEnv {

  case class Record(key: String, size: Int, record: Int, date: String)

  lazy val _data = sqlc.createDataFrame(
    Map("small" -> 15, "large" -> 120, "medium" -> 50).flatMap(x => {
      for (
        i <- 0 until x._2;
        date <- Seq("2015-01-27", "2015-01-28", "2015-01-29", "2015-01-30", "2015-02-01", "2015-02-02"))
        yield Record(
          key = x._1,
          size = x._2,
          record = i,
          date = date
        )
    }).toSeq)

  lazy val _transformed = new OrderedCutEstimator()
    .setGroupByColumns("key")
    .setSortByColumn("date")
    .setExpectedRecords(300)
    .setDescending(true)
    .fit(_data)
    .transform(_data)

  lazy val _transformedAscending = new OrderedCutEstimator()
    .setGroupByColumns("key")
    .setSortByColumn("date")
    .setExpectedRecords(300)
    .fit(_data)
    .transform(_data)

  lazy val _transformedMultiGroup = new OrderedCutEstimator()
    .setGroupByColumns("key", "size")
    .setSortByColumn("date")
    .setExpectedRecords(300)
    .fit(_data)
    .transform(_data)
}
