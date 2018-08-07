package org.apache.spark.sql.odkl

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.DataType


/**
 * Proxy class that allows to access private functionality from package `org.apache.spark.sql`.
 *
 * Created by vyacheslav.baranov on 14/07/15.
 */
object SparkSqlUtils {

  def reflectionLock: AnyRef = new Object

  def customUDF(f: AnyRef,
              dataType: DataType,
              inputTypes: Option[Seq[DataType]]) : UserDefinedFunction
  = UserDefinedFunction(f, dataType, inputTypes)

}
