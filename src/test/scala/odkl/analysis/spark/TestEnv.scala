package odkl.analysis.spark

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by vyacheslav.baranov on 22/12/15.
 */
trait TestEnv {

  lazy val sparkConf = TestEnv._sparkConf
  lazy val sc = TestEnv._sc
  lazy val sqlc = TestEnv._sqlc
}

object TestEnv extends TestEnv {

  System.setProperty("hadoop.home.dir", "/")

  private lazy val _sparkConf = new SparkConf()
    .setMaster("local[4]")
    .setAppName(getClass.getName)
    .set("spark.sql.shuffle.partitions", "2")

  private lazy val _sc = new SparkContext(_sparkConf)

  private lazy val _sqlc = new SQLContext(_sc)

}
