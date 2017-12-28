package org.apache.spark.sql.odkl

import org.apache.spark.sql.catalyst.ScalaReflectionLock


/**
 * Proxy class that allows to access private functionality from package `org.apache.spark.sql`.
 *
 * Created by vyacheslav.baranov on 14/07/15.
 */
object SparkSqlUtils {

  def reflectionLock: AnyRef = ScalaReflectionLock

}
