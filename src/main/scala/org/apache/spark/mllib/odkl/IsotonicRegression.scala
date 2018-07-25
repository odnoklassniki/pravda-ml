package org.apache.spark.mllib.odkl

import java.io.Serializable
import java.lang.{Double => JDouble}

import odkl.analysis.spark.util.Logging
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.regression.IsotonicRegressionModel
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Isotonic regression.
  * Currently implemented using parallelized pool adjacent violators algorithm.
  * Only univariate (single feature) algorithm supported.
  *
  * Sequential PAV implementation based on:
  * Tibshirani, Ryan J., Holger Hoefling, and Robert Tibshirani.
  *   "Nearly-isotonic regression." Technometrics 53.1 (2011): 54-61.
  *   Available from [[http://www.stat.cmu.edu/~ryantibs/papers/neariso.pdf]]
  *
  * Sequential PAV parallelization based on:
  * Kearsley, Anthony J., Richard A. Tapia, and Michael W. Trosset.
  *   "An approach to parallelizing isotonic regression."
  *   Applied Mathematics and Parallel Computing. Physica-Verlag HD, 1996. 141-147.
  *   Available from [[http://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR96640.pdf]]
  *
  * @see [[http://en.wikipedia.org/wiki/Isotonic_regression Isotonic regression (Wikipedia)]]
  *
  * ODKL Patches:
  *   1. view before slice for better performance
  *   2. Per partition sorting instead of global order for smother data distribution
  */
@Since("1.3.0")
class IsotonicRegression private (private var isotonic: Boolean) extends Serializable with Logging{

  /**
    * Constructs IsotonicRegression instance with default parameter isotonic = true.
    *
    * @return New instance of IsotonicRegression.
    */
  @Since("1.3.0")
  def this() = this(true)

  /**
    * Sets the isotonic parameter.
    *
    * @param isotonic Isotonic (increasing) or antitonic (decreasing) sequence.
    * @return This instance of IsotonicRegression.
    */
  @Since("1.3.0")
  def setIsotonic(isotonic: Boolean): this.type = {
    this.isotonic = isotonic
    this
  }

  /**
    * Run IsotonicRegression algorithm to obtain isotonic regression model.
    *
    * @param input RDD of tuples (label, feature, weight) where label is dependent variable
    *              for which we calculate isotonic regression, feature is independent variable
    *              and weight represents number of measures with default 1.
    *              If multiple labels share the same feature value then they are ordered before
    *              the algorithm is executed.
    * @return Isotonic regression model.
    */
  @Since("1.3.0")
  def run(input: RDD[(Double, Double, Double)]): IsotonicRegressionModel = {
    val preprocessedInput = if (isotonic) {
      input
    } else {
      input.map(x => (-x._1, x._2, x._3))
    }

    val pooled = parallelPoolAdjacentViolators(preprocessedInput)

    val predictions: Array[Double] = if (isotonic) pooled.map(_._1) else pooled.map(-_._1)
    val boundaries: Array[Double] = pooled.map(_._2)

    if (predictions(0).isNaN) {
      logWarning("Got NaN at the beginning of the predictions, replacing with 0.0")
      predictions(0) = 0.0
    }

    if (boundaries(0).isNaN) {
      logWarning("Got NaN at the beginning of the boundaries, replacing with 0.0")
      boundaries(0) = 0.0
    }

    new IsotonicRegressionModel(boundaries, predictions, isotonic)
  }

  /**
    * Run pool adjacent violators algorithm to obtain isotonic regression model.
    *
    * @param input JavaRDD of tuples (label, feature, weight) where label is dependent variable
    *              for which we calculate isotonic regression, feature is independent variable
    *              and weight represents number of measures with default 1.
    *              If multiple labels share the same feature value then they are ordered before
    *              the algorithm is executed.
    * @return Isotonic regression model.
    */
  @Since("1.3.0")
  def run(input: JavaRDD[(JDouble, JDouble, JDouble)]): IsotonicRegressionModel = {
    run(input.rdd.retag.asInstanceOf[RDD[(Double, Double, Double)]])
  }

  /**
    * Performs a pool adjacent violators algorithm (PAV).
    * Uses approach with single processing of data where violators
    * in previously processed data created by pooling are fixed immediately.
    * Uses optimization of discovering monotonicity violating sequences (blocks).
    *
    * @param input Input data of tuples (label, feature, weight).
    * @return Result tuples (label, feature, weight) where labels were updated
    *         to form a monotone sequence as per isotonic regression definition.
    */
  private def poolAdjacentViolators(
                                     input: Array[(Double, Double, Double)]): Array[(Double, Double, Double)] = {

    if (input.isEmpty) {
      return Array.empty
    }

    // Pools sub array within given bounds assigning weighted average value to all elements.
    def pool(input: Array[(Double, Double, Double)], start: Int, end: Int): Unit = {
      var weightedSum: Double = 0.0
      var weight: Double = 0.0

      input.view.slice(start, end + 1).foreach(x => {
        weightedSum = weightedSum + x._1 * x._3
        weight = weight + x._3
      })

      val update = if (weight != 0.0) weightedSum / weight else 0.0

      var i = start
      while (i <= end) {
        input(i) = (update, input(i)._2, input(i)._3)
        i = i + 1
      }
    }

    var i = 0
    val len = input.length
    while (i < len) {
      var j = i

      // Find monotonicity violating sequence, if any.
      while (j < len - 1 && input(j)._1 > input(j + 1)._1) {
        j = j + 1
      }

      // If monotonicity was not violated, move to next data point.
      if (i == j) {
        i = i + 1
      } else {
        // Otherwise pool the violating sequence
        // and check if pooling caused monotonicity violation in previously processed points.
        while (i >= 0 && input(i)._1 > input(i + 1)._1) {
          pool(input, i, j)
          i = i - 1
        }

        i = j
      }
    }

    // For points having the same prediction, we only keep two boundary points.
    val compressed = ArrayBuffer.empty[(Double, Double, Double)]

    var (curLabel, curFeature, curWeight) = input.head
    var rightBound = curFeature
    def merge(): Unit = {
      compressed += ((curLabel, curFeature, curWeight))
      if (rightBound > curFeature) {
        compressed += ((curLabel, rightBound, 0.0))
      }
    }
    i = 1
    while (i < input.length) {
      val (label, feature, weight) = input(i)
      if (label == curLabel) {
        curWeight += weight
        rightBound = feature
      } else {
        merge()
        curLabel = label
        curFeature = feature
        curWeight = weight
        rightBound = curFeature
      }
      i += 1
    }
    merge()

    compressed.toArray
  }

  /**
    * Performs parallel pool adjacent violators algorithm.
    * Performs Pool adjacent violators algorithm on each partition and then again on the result.
    *
    * @param input Input data of tuples (label, feature, weight).
    * @return Result tuples (label, feature, weight) where labels were updated
    *         to form a monotone sequence as per isotonic regression definition.
    */
  private def parallelPoolAdjacentViolators(
                                             input: RDD[(Double, Double, Double)]): Array[(Double, Double, Double)] = {

    // This is the original version. Implying global sort we easily can get a very imbalanced partitioning
    // Thus in ODKL Patch we replace it with per-partition sort and it produces calibration of the same quality
    // without a risk of stucking on imbalanced partition.
//    val parallelStepResult = input
//      .sortBy(x => (x._2, x._1))
//      .glom()
//      .flatMap(poolAdjacentViolators)
//      .collect()
//      .sortBy(x => (x._2, x._1)) // Sort again because collect() doesn't promise ordering.
//    poolAdjacentViolators(parallelStepResult)

    val parallelStepResult = input
      .glom()
      .flatMap(arr => {
        poolAdjacentViolators(arr.sortBy(x => (x._2, x._1)))
      })
      .collect()
      .sortBy(x => (x._2, x._1)) // Sort again because collect() doesn't promise ordering.
    poolAdjacentViolators(parallelStepResult)
  }
}

