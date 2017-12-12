package odkl.analysis.spark.util

import odkl.analysis.spark.TestEnv
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.scalatest.{FlatSpec, Matchers}

/**
 * Created by vyacheslav.baranov on 02/12/15.
 */
class SQLOperationsSpec extends FlatSpec with TestEnv with Matchers with SQLOperations {

  import sqlc.implicits._

  "A CollectAsList" should "aggregate items" in {
    val df = sc.parallelize(Seq(
      "A" -> 10,
      "A" -> 11,
      "A" -> 12,
      "B" -> 20,
      "B" -> 21,
      "C" -> 30
    )).toDF("C1", "C2")
    val res = df.groupBy("C1").agg(collectAsList(IntegerType)(col("C2"))).collect()
    assertResult(3)(res.length)
    val c1 = res.find(_.getString(0) == "A").getOrElse(fail("No row for 'A'")).getAs[Seq[Int]](1)
    c1 should contain theSameElementsAs Seq(10, 11, 12)
    val c2 = res.find(_.getString(0) == "B").getOrElse(fail("No row for 'B'")).getAs[Seq[Int]](1)
    c2 should contain theSameElementsAs Seq(20, 21)
    val c3 = res.find(_.getString(0) == "C").getOrElse(fail("No row for 'C'")).getAs[Seq[Int]](1)
    c3 should contain theSameElementsAs Seq(30)
  }
}
