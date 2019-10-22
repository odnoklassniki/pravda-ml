package odkl.analysis.spark

import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by vyacheslav.baranov on 22/12/15.
 */
trait TestEnv {

  lazy val sparkConf = TestEnv._sparkConf
  lazy val spark = TestEnv._spark
  lazy val sc = TestEnv._sc
  lazy val sqlc = TestEnv._sqlc
}

object TestEnv extends TestEnv {

  System.setProperty("hadoop.home.dir", "/")

  private lazy val _sparkConf = new SparkConf()
    .setMaster("local[4]")
    .setAppName(getClass.getName)
    .set("spark.sql.shuffle.partitions", "2")

  private lazy val _spark = SparkSession.builder().config(_sparkConf).getOrCreate()

  private lazy val _sc = _spark.sparkContext

  private lazy val _sqlc = _spark.sqlContext

}
