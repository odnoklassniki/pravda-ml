package org.apache.spark.ml.odkl

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalatest.FlatSpec

class QuantileDiscretizerSpec extends FlatSpec with TestEnv with org.scalatest.Matchers
  with SQLOperations {

  private lazy val data = QuantileDiscretizerSpec._data
  private lazy val model = QuantileDiscretizerSpec._model
  private lazy val transformed = QuantileDiscretizerSpec._transformed

  "QuantileDiscretize" should "found 10 buckets for filled column" in {
    val fullSplits = model.getSplitsArray(0)

    fullSplits should contain theSameElementsInOrderAs (Seq(Double.NegativeInfinity) ++
      Array.tabulate(10) { i => Math.pow(10, i) } ++
      Seq(Double.PositiveInfinity))
  }

  "QuantileDiscretize" should "found 5 buckets for partly filled column" in {
    val fullSplits = model.getSplitsArray(1)

    fullSplits should contain theSameElementsInOrderAs (Seq(Double.NegativeInfinity) ++
      Array.tabulate(5) { _ + 1.0 } ++
      Seq(Double.PositiveInfinity))
  }

  "QuantileDiscretize" should "found 1 bucket for partly filled column" in {
    val fullSplits = model.getSplitsArray(2)

    fullSplits should contain theSameElementsInOrderAs Seq(Double.NegativeInfinity, 1.12, Double.PositiveInfinity)
  }

  "QuantileDiscretize" should "add zero bucket for empty column" in {
    val fullSplits = model.getSplitsArray(3)

    fullSplits should contain theSameElementsInOrderAs Seq(Double.NegativeInfinity, 0.0, Double.PositiveInfinity)
  }

  import sqlc.implicits._

  "Transformed data" should "contain only valid buckets for full column" in {
    val values = transformed.select('full_bucket.as[Double]).distinct().collect().sorted
    values should contain theSameElementsInOrderAs Array.tabulate(10){_ + 1.0}
  }

  "Transformed data" should "contain only valid buckets for partly filled column" in {
    val values = transformed.select('partlyEmpty_bucket.as[Option[Double]]).distinct().collect().sorted
    values should contain theSameElementsInOrderAs Seq(None) ++ Array.tabulate(5){i => Some(i + 1.0)}
  }

  "Transformed data" should "contain only single buckets for constant column" in {
    val values = transformed.select('constant_bucket.as[Double]).distinct().collect().sorted
    values should contain theSameElementsInOrderAs Seq(1.0)
  }

  "Transformed data" should "contain single buckets for empty column" in {
    val values = transformed.select('empty_bucket.as[Option[Double]]).distinct().collect().sorted
    values should contain theSameElementsInOrderAs Seq(None)
  }
}

object QuantileDiscretizerSpec extends TestEnv {
  import sqlc.sparkSession.implicits._

  case class Entry(full: Double, partlyEmpty: Option[Double], constant: Double = 1.12, empty: Option[Double] = None)

  private val entries = Seq(
    Entry(1, Some(1.0)),
    Entry(10, Some(2.0)),
    Entry(100, Some(3.0)),
    Entry(1000, Some(4.0)),
    Entry(10000, Some(5.0)),
    Entry(100000, None),
    Entry(1000000, None),
    Entry(10000000, None),
    Entry(100000000, None),
    Entry(1000000000, None)
  )
  lazy val _data: Dataset[Entry] = (entries ++ entries ++ entries ++ entries).toDS

  lazy val _model: Bucketizer = new QuantileDiscretizer()
    .setNumBuckets(20)
    .setInputCols(_data.schema.fieldNames)
    .setOutputCols(_data.schema.fieldNames.map(_ + "_bucket"))
    .fit(_data)

  lazy val _transformed: DataFrame = _model.transform(_data)
  
}