package org.apache.spark.ml.odkl

import org.apache.spark.ml.param.{IntParam, Params}

trait HasNumThreads {
  this: Params =>

  final val numThreads = new IntParam(this, "numThreads", "How many threads to use for fitting forks.")

  def setNumThreads(value: Int): this.type = set(numThreads, value)
}
