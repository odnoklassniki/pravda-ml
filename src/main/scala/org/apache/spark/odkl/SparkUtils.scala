package org.apache.spark.odkl

import org.apache.spark.rdd.RDDOperationScope
import org.apache.spark.util.ClosureCleaner
import org.apache.spark.SparkContext

/**
 * Proxy class that allows to access private functionality from package `org.apache.spark`.
 *
 * Created by vyacheslav.baranov on 17/06/15.
 */
object SparkUtils {

  /**
   * Cleans closure of a function of unused/non-serializable objects.
   *
   * @see [[org.apache.spark.SparkContext.clean]]
   * @param f
   * @param checkSerializable
   * @tparam F
   * @return
   */
  def clean[F <: AnyRef](f: F, checkSerializable: Boolean = true): F = {
    ClosureCleaner.clean(f, checkSerializable)
    f
  }

  /**
   * Used to improve readability of a stage in Spark Web UI.
   *
   * @see [[org.apache.spark.rdd.RDD.withScope]]
   * @param sc
   * @param body
   * @tparam U
   * @return
   */
  def withScope[U](sc: SparkContext, allowNesting: Boolean = false)(body: => U): U = {
    RDDOperationScope.withScope[U](sc, allowNesting)(body)
  }

  /**
   * Returns the classloader of current thread or the classloader that loaded this jar
   *
   * @return
   */
  def getContextClassLoader: ClassLoader = {
    Option(Thread.currentThread().getContextClassLoader).getOrElse(getClass.getClassLoader)
  }

}
