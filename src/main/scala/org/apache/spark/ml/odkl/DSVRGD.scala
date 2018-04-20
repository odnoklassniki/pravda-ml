package org.apache.spark.ml.odkl

import odkl.analysis.spark.util.collection.CompactBuffer
import org.apache.spark.SparkContext
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.odkl.DSVRGD.{DistributedSgdState, LossRecord}
import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable, SchemaUtils}
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, functions}

import scala.util.Random

/**
  * Created by dmitriybugaichenko on 10.11.16.
  *
  * Implementation of a distributed version of Stochastic Variance Reduced Gradient Descent. The idea
  * is taken from https://arxiv.org/abs/1512.01708 - input dataset is partitioned and workers performs
  * descent simultaneously updating own copy of the weights at each random point (following SGD schema).
  * At the end of epoche data from all workers are collected and aggregated. Variance reduction is
  * achieved by keeping average gradient from previous iterations and evaluating gradient at one extra point
  * (average of all weights seen during previous epoche). The update rule is:
  *
  * w_new = w_old − η (∇f_i(w_old) − ∇f_i(w_avg) + g)
  *
  * TODO: Other variance reduction and step size tuning techniques might be applied.
  *
  * Requires AttributeGroup metadata for both labels and features, supports elastic net regularization and
  * multiple parallel labels training (similar to MatrixLBFGS).
  */
abstract class DSVRGD[M <: ModelWithSummary[M]]
(
  override val uid: String
) extends SummarizableEstimator[M]
  with HasPredictionCol with HasFeaturesCol with HasLabelCol
  with HasRegParam with HasElasticNetParam with HasNetlibBlas
  with HasMaxIter with HasTol with HasCacheTrainData {


  // TODO: Consider other learning rate strategies (may be plugable). Tried Adam and AdaDelta without success.
  // The sklearn way: http://leon.bottou.org/publications/pdf/mloptbook-2011.pdf
  val learningRate = new DoubleParam(this, "learningRate", "Speed of update. Might be decreased if loss functions degrades or increased if the loss keeps decreasing.")

  val convergenceMode = new Param[String](
    this, "convergenceMode", "Defines how to check for convergence: weights, loss, both, any",
    Set("weights", "loss", "both", "any"))

  val lastIsIntercept = new BooleanParam(
    this, "lastIsIntercept", "Whenever to treat the last feature as intercept (should not be regularized and properly initialized).")

  val localMinibatchSize = new Param[Int](this, "localMinibatchSize",
    "Amount of samples to group into mini-batches localy when computing gradient. Makes gradient approximation more preciese.",
    (x: Int) => x > 0 && x < 100000)

  val lossIncreaseTolerance = new DoubleParam(this, "lossIncreaseTolerance",
    "Maximum allowed relative increase of the loss function. If we go beyond that, decrease the learning rate.")

  val speedUpFactor = new DoubleParam(this, "speedUpFactor",
    "Percentage of learning rate increase for cases when we keep moving in the right direction (loss decreases).")

  val slowDownFactor = new DoubleParam(this, "slowDownFactor",
    "Percentage of learning rate decrease for cases when we moving in a wrong direction (loss increases).")

  setDefault(
    regParam -> 0.0,
    elasticNetParam -> 0.0,
    learningRate -> 1.0,
    maxIter -> 100,
    lastIsIntercept -> false,
    tol -> 0.001,
    cacheTrainData -> true,
    convergenceMode -> "both",
    localMinibatchSize -> 50,
    lossIncreaseTolerance -> 1e-10,
    speedUpFactor -> 0.01,
    slowDownFactor -> 0.9
  )

  def setRegParam(value: Double): this.type = set(regParam, value)

  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)

  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setLastIsIntercept(value: Boolean): this.type = set(lastIsIntercept, value)

  def setTol(value: Double): this.type = set(tol, value)

  def setConvergenceMode(value: String): this.type = set(convergenceMode, value)

  def setLocalMinibatchSize(value: Int): this.type = set(localMinibatchSize, value)

  def setSpeedUpFactor(value: Double) : this.type = set(speedUpFactor, value)
  def setSlowDownFactor(value: Double) : this.type = set(slowDownFactor, value)

  override def fit(dataset: Dataset[_]): M = {
    // Extract information regarding labels and features.
    val labelColumn = dataset.schema($(labelCol))
    val labelAttributeGroup = AttributeGroup.fromStructField(labelColumn)
    val numLabels = labelAttributeGroup.size
    val numFeatures = AttributeGroup.fromStructField(dataset.schema($(featuresCol))).size

    val labelNames: Array[String] = labelAttributeGroup.attributes
      .map(_.map(a => a.name.getOrElse(a.index.get.toString)))
      .getOrElse(Array.tabulate(numLabels) { i => i.toString })

    val sc = dataset.sqlContext.sparkContext

    // Check if we need to cache data
    val (data, cleaner) = if ($(cacheTrainData)) {
      (dataset.select($(featuresCol), $(labelCol)).cache(), (x: DataFrame) => x.unpersist())
    } else {
      (dataset.select($(featuresCol), $(labelCol)), (x: DataFrame) => {})
    }

    // Prepare state data
    var activeLabels = Array.tabulate(numLabels) { i => i }
    var avgWeights: Matrix = new SparseMatrix(numLabels, numFeatures, new Array[Int](numFeatures + 1), new Array[Int](0), new Array[Double](0))
    var avgGradient: Matrix = avgWeights.copy
    var weights: Matrix = initializeWeights(data, numLabels, numFeatures)

    // Number of current epoch
    var step = 1

    // Store loss history here in order to check for convergence and add it to summary
    // TODO: Cosnider evaluating and storing extra metrics (eg. http://chbrown.github.io/kdd-2013-usb/kdd/p1294.pdf)
    val lossHistory: Array[CompactBuffer[Double]] = Array.tabulate(numLabels) { i => new CompactBuffer[Double]() }
    val weightDiffHistory: Array[CompactBuffer[Double]] = Array.tabulate(numLabels) { i => new CompactBuffer[Double]() }
    val weightNormHistory: Array[CompactBuffer[Double]] = Array.tabulate(numLabels) { i => new CompactBuffer[Double]() }

    try {
      // Prepare regularization params vectors
      val l1Scalar = $(regParam) * $(elasticNetParam)
      val l1RegParam = if (l1Scalar > 0) evaluateL1Regularization(data, l1Scalar, numLabels) else null.asInstanceOf[Vector]

      val l2Scalar = $(regParam) * (1.0 - $(elasticNetParam))
      val l2RegParam = if (l2Scalar > 0) evaluateL2Regularization(data, l2Scalar, numLabels) else null.asInstanceOf[Vector]

      val skipRegFeature = if ($(lastIsIntercept)) numFeatures - 1 else -1

      val learningRates = new DenseVector(new Array[Double](numLabels))
      learningRates.values.transform(x => $(learningRate))

      do {
        // Broadcast matrices involved
        val weightsBroadcast = sc.broadcast(relabelMatrix(activeLabels, weights))
        val avgWeightsBroadcast = sc.broadcast(relabelMatrix(activeLabels, avgWeights))
        val avgGradientBroadcast = sc.broadcast(relabelMatrix(activeLabels, avgGradient))

        // Do the step
        val state = try {
          singleStep(
            data.toDF.rdd.map(r => {
              r.getAs[Vector](0) -> relabel(activeLabels, r.getAs[Vector](1))
            }),
            weightsBroadcast,
            avgWeightsBroadcast,
            avgGradientBroadcast,
            relabel(activeLabels, l1RegParam),
            relabel(activeLabels, l2RegParam),
            step,
            relabel(activeLabels,learningRates)
          )
        } finally {
          weightsBroadcast.destroy()
          avgWeightsBroadcast.destroy()
          avgGradientBroadcast.destroy()
        }

        val degradedSet = new scala.collection.mutable.HashSet[Int]()
        val nanSet = new scala.collection.mutable.HashSet[Int]()
        val activeMap = new scala.collection.mutable.HashMap[Int,Int]()


        // Store loss
        state.accumulatedLoss.foreachActive((index, loss) => {
          val label = activeLabels(index)
          val history = lossHistory(label)
          val prevLost = history.lastOption.getOrElse(loss)
          history += loss

          if (loss.isNaN) {
            logError(s"Got NaN loss for ${labelNames(label)}, giving up")
            nanSet += label
          } else if (loss > prevLost + Math.abs(prevLost * $(lossIncreaseTolerance))) {
            val prevLearningRate = learningRates(label)
            learningRates.values(label) *= $(slowDownFactor)

            degradedSet += label

            logInfo(s"Got increase in loss function for ${labelNames(label)}, slowing down from $prevLearningRate to ${learningRates(label)}")
          } else {
            activeMap.put(label, index)
            if(loss < prevLost) {
              // We are going fine, increase learning rate
              learningRates.values(label) *= 1.0 + $(speedUpFactor)
            }
          }
        })

        // Merge converged and still active labels data
        val newWeights = merge(activeLabels.zipWithIndex.toMap, weights, state.weights)
        avgWeights = merge(activeMap.toMap, avgWeights, state.accumulatedWeights)
        avgGradient = merge(activeMap.toMap, avgGradient, state.accumulatedGradient)

        if (l1RegParam != null) {
          applyL1Shrinkage(l1RegParam, newWeights, skipRegFeature, activeMap.keySet.toSet)
          applyL1Shrinkage(l1RegParam, avgWeights.asInstanceOf[DenseMatrix], skipRegFeature, activeMap.keySet.toSet)
          MatrixUtils.applyAll(
            newWeights,
            avgGradient.asInstanceOf[DenseMatrix],
            (label, feature, xv, v) => {
              if (feature == skipRegFeature || !activeMap.contains(label)) {
                v
              } else {
                val l1regValue = l1RegParam(label)

                // This part is taken from Breeze.OWLQN
                xv match {
                  case 0.0 => {
                    val delta_+ = v + l1regValue
                    val delta_- = v - l1regValue
                    if (delta_- > 0) delta_- else if (delta_+ < 0) delta_+ else 0.0
                  }
                  case _ => v // + Math.signum(xv) * l1regValue
                }
              }
            }
          )
        }

        // Get stat for weights diff and norm
        activeLabels.foreach(label => {
          weightDiffHistory(label) += weightsDistanceForLabel(weights, newWeights, label)
          weightNormHistory(label) += weightNorm(newWeights, label, skipRegFeature)
        })

        logInfo(s"Done iteration $step," +
          s" loses ${activeLabels.map(label => labelNames(label) -> lossHistory(label).takeRight(3)).toMap},\n" +
          s" diffs ${activeLabels.map(label => labelNames(label) -> weightDiffHistory(label).takeRight(3)).toMap},\n" +
          s" norms ${activeLabels.map(label => labelNames(label) -> weightNormHistory(label).takeRight(3)).toMap}")

        // Extract labels still not converged
        activeLabels = (getNotConverged(activeMap.toMap, lossHistory, weightDiffHistory, weightNormHistory, $(tol)) ++ degradedSet)
            .sorted

        // Go next step
        weights = merge(activeMap.toMap, weights, state.weights)
        step += 1
      } while (activeLabels.length > 0 && step <= $(maxIter))

      // Check if we done due convergence or by spinning too long.
      if (step < $(maxIter)) {
        logInfo(s"All labels converged in $step iterations.")
      } else {
        logWarning(s"Failed to converge in ${$(maxIter)} iterations.")
      }

    } finally {
      // Unpresist data iw we cached it
      cleaner(data)
    }

    // Create an appropriate model
    val model = extractModel(labelAttributeGroup, numLabels, weights, dataset.toDF)
      .setParent(this)

    // Add summary info with loss history
    model.setSummary(model.summary.copy(extractSummaryBlocks(lossHistory, weightDiffHistory, weightNormHistory, dataset.toDF, labelAttributeGroup)))
  }

  /**
    * Extracts summary blocks from iterations loss history.
    */
  protected def extractSummaryBlocks(
                                      lossHistory: Array[CompactBuffer[Double]],
                                      weightDiffHistory: Array[CompactBuffer[Double]],
                                      weightNormHistory: Array[CompactBuffer[Double]],
                                      dataset: DataFrame,
                                      labelAttributeGroup: AttributeGroup): Map[ModelWithSummary.Block, DataFrame] = {
    val numLabels = labelAttributeGroup.size

    val names: Map[Int, String] = labelAttributeGroup.attributes
      .map(x => x.map(a => {
        val index: Int = a.index.get
        index -> a.name.getOrElse(index.toString)
      }).toMap)
      .getOrElse(Array.tabulate(numLabels) { i => i -> i.toString }.toMap)

    val sc = dataset.sqlContext.sparkContext

    Map(
      DSVRGD.lossHistory -> extractBlock(lossHistory, dataset, names, sc),
      DSVRGD.WeightDiffHistory -> extractBlock(weightDiffHistory, dataset, names, sc),
      DSVRGD.WeightNormHistory -> extractBlock(weightNormHistory, dataset, names, sc))
  }

  def extractBlock(lossHistory: Array[CompactBuffer[Double]], dataset: DataFrame, names: Map[Int, String], sc: SparkContext): DataFrame = {
    dataset.sqlContext.createDataFrame(
      sc.parallelize(
        lossHistory.zipWithIndex.flatMap(x => x._1.zipWithIndex.map(y => LossRecord(names(x._2), y._2, y._1))).toSeq,
        1))
      .withColumnRenamed("label", $(labelCol))
  }

  /**
    * Given labels info and weights matrice create appropriate ML models.
    */
  protected def extractModel(labelAttributeGroup: AttributeGroup, numLabels: Int, weights: Matrix, dataset: DataFrame): M

  def initializeWeights(data: DataFrame, numLabels: Int, numFeatures: Int): Matrix =
    new SparseMatrix(numLabels, numFeatures, new Array[Int](numFeatures + 1), new Array[Int](0), new Array[Double](0))

  /**
    * Given L2 regularization config create a vector with per-label reg param (by default - constant).
    */
  protected def evaluateL2Regularization(data: DataFrame, l2Scalar: Double, numLabels: Int): Vector =
    Vectors.dense(Array.fill(numLabels)(l2Scalar))

  /**
    * Given L1 regularization config create a vector with per-label reg param (by default - constant).
    */
  protected def evaluateL1Regularization(data: DataFrame, l1Scalar: Double, numLabels: Int): Vector =
    Vectors.dense(Array.fill(numLabels)(l1Scalar))

  /**
    * Utility used to split weights matrice into label -> vector map
    */
  protected def extractLabelVectors(labelAttributeGroup: AttributeGroup, numLabels: Int, weights: Matrix): Map[String, Vector] = {
    val attributes: Array[Attribute] = labelAttributeGroup.attributes
      .getOrElse(Array.tabulate(numLabels) { i => NumericAttribute.defaultAttr.withIndex(i) })

    val result = Array.tabulate(numLabels) { label =>
      attributes(label).name.getOrElse(label.toString) -> extractRow(label, weights)
    }.toMap
    result
  }

  /**
    * Extracts a single row from a matrice.
    */
  protected def extractRow(label: Int, weights: Matrix): Vector = {
    Vectors.dense(Array.tabulate(weights.numCols) { feature => weights(label, feature) }).compressed
  }

  /**
    * Used to preserve only active (not yet converged) labels into a vector
    */
  protected def relabel(activeLabels: Array[Int], labels: Vector): DenseVector = {
    if (labels == null) {
      null
    } else if (activeLabels.length == labels.size) {
      labels.toDense
    } else {
      new DenseVector(activeLabels.map(labels(_)))
    }
  }

  /**
    * Used to preserve only active (not yet converged) labels into a matrix
    */
  protected def relabelMatrix(activeLabels: Array[Int], matrix: Matrix): Matrix = {
    if (activeLabels.length == matrix.numRows) {
      matrix
    } else {
      MatrixUtils.transformDense(
        DenseMatrix.zeros(activeLabels.length, matrix.numCols),
        (label, feature, weight) => matrix(activeLabels(label), feature)
      )
    }
  }

  /**
    * Single epoch of the descend
    *
    * @param data        Data with features and labels
    * @param weights     Weghts matrix to start with.
    * @param avgWeights  Average weights among walked during previous epoch.
    * @param avgGradient Average gradient among seen during previous epoch.
    * @param l1regParam  Vector with the strength of L1 regularization (null if disabled)
    * @param l2regParam  Vector with the strength of L2 regularization (null if disabled)
    * @param stepNum     Number of epoch
    * @return State with weights, averages and loss from this epoch
    */
  protected def singleStep(
                            data: RDD[(Vector, DenseVector)],
                            weights: Broadcast[Matrix],
                            avgWeights: Broadcast[Matrix],
                            avgGradient: Broadcast[Matrix],
                            l1regParam: Vector,
                            l2regParam: Vector,
                            stepNum: Int,
                            labelLearningRates: DenseVector): DistributedSgdState = {

    val doRegularizeLast = !$(lastIsIntercept)

    val batchSize = $(localMinibatchSize)

    // Do the descend in parallel for each partition
    data.mapPartitions(iter => {

      // Prepare all the local data
      var count = 0

      val localWeights = toDense(weights)

      val lossCache = Vectors.zeros(localWeights.numRows).toDense
      val accumulatedLoss = Vectors.zeros(localWeights.numRows).toDense
      val accumulatedLossCompensator = accumulatedLoss.copy
      val tLoss = accumulatedLoss.copy
      val yLoss = accumulatedLoss.copy
      val updateTerm = DenseMatrix.zeros(localWeights.numRows, localWeights.numCols)
      val accumulatedGradient = updateTerm.copy
      val accumulatedGradientCompensator = updateTerm.copy
      val accumulatedWeights = updateTerm.copy
      val accumulatedWeightsCompensator = updateTerm.copy
      val t = updateTerm.copy
      val y = updateTerm.copy

      val skipRegFeature = if (doRegularizeLast) -1 else localWeights.numCols - 1

      val averageGrad = toDense(avgGradient)
      val averageWeights = toDense(avgWeights)

      val margins = new Array[Double](batchSize * localWeights.numRows)
      val samples = new Array[Double](batchSize * localWeights.numCols)
      val labels = new Array[Double](batchSize * localWeights.numRows)
      var batchedSamples = 0
      var sampleShift = 0
      var labelsShift = 0

      val learningRates = DenseMatrix.zeros(localWeights.numRows, localWeights.numCols)
      MatrixUtils.transformDense(
        learningRates,
        (label, feature, rate) => labelLearningRates(label))


      // Note that here we materialize all the data in order to enforce random walking order.
      // This is because we expect DSVRGD is used in combination with sampler or with data split
      // into small enougth partitions
      def minibatchStep: Unit = {
        val batchedFeatures = new DenseMatrix(localWeights.numCols, batchedSamples, samples)
        val batchedLabels = new DenseMatrix(localWeights.numRows, batchedSamples, labels)
        val marginCache = new DenseMatrix(localWeights.numRows, batchedSamples, margins)

        // Gradient at current weights
        fullGradientAndLoss(l1regParam, l2regParam, localWeights, marginCache, lossCache, updateTerm, skipRegFeature, batchedFeatures, batchedLabels)

        // Keep track of all gradients and loss we've seen
        axpyCompensated(lossCache.values, accumulatedLoss.values, accumulatedLossCompensator.values, yLoss.values, tLoss.values)
        axpyCompensated(updateTerm.values, accumulatedGradient.values, accumulatedGradientCompensator.values, y.values, t.values)

        // Adjust for own gradient
        adjust(-1, learningRates, updateTerm, localWeights)

        // First epoche is for exploration, there are no averages
        if (stepNum > 1) {
          // Compute gradient at average weights point (with negative weight)
          fullGradientAndLoss(l1regParam, l2regParam, averageWeights, marginCache, lossCache, updateTerm, skipRegFeature, batchedFeatures, batchedLabels)

          // Adjust for average weights gradient
          adjust(1, learningRates, updateTerm, localWeights)

          // Adjust for average gradient
          adjust(-1, learningRates, averageGrad, localWeights)
        }

        // Memorize all the weights we walked through
        axpyCompensated(localWeights.values, accumulatedWeights.values, accumulatedWeightsCompensator.values, y.values, t.values)

        // Keep track of the number of items
        count += 1

        // Reset batch size
        batchedSamples = 0
        sampleShift = 0
        labelsShift = 0
      }

      for ((sample, label) <- Random.shuffle(iter)) {

        sample.foreachActive((i, w) => samples(sampleShift + i) = w)
        label.foreachActive((i, w) => labels(labelsShift + i) = w)
        sampleShift += localWeights.numCols
        labelsShift += localWeights.numRows
        batchedSamples += 1

        if (batchedSamples >= batchSize) {
          minibatchStep
        }
      }

      // Yes, ignore the last mini-batch. It is smaller and thus less precise, but last step is the most
      // important.
      //if (batchedSamples > 0) {
      //  minibatchStep
      //}

      if (count > 0) {
        val divider = 1.0 / count

        dscal(count, localWeights.values)

        // Produce one record per partition
        Iterator(DistributedSgdState(localWeights, accumulatedGradient, accumulatedWeights, accumulatedLoss, count))
      } else {
        Iterator()
      }
    })
      // Aggregate and scale the overall result
      .treeReduce((x, y) => x.merge(y)).scale()
  }

  def toDense(weights: Broadcast[Matrix]): DenseMatrix = {
    weights.value match {
      case d: DenseMatrix => d.copy
      case s: SparseMatrix => s.toDense
    }
  }

  def adjust(direction: Int, learningRates: DenseMatrix, updateTerm: DenseMatrix, weights: DenseMatrix) = {
    MatrixUtils.applyNonZeros(
      learningRates,
      weights,
      (label, feature, rate, weight) => weight + direction * rate * updateTerm(label, feature)
    )
  }

  def fullGradientAndLoss(
                           l1regParam: Vector,
                           l2regParam: Vector,
                           localWeights: DenseMatrix,
                           marginCache: DenseMatrix,
                           lossCache: DenseVector,
                           updateTerm: DenseMatrix,
                           skipRegFeature: Int,
                           features: DenseMatrix,
                           labels: DenseMatrix): Any = {
    // Reset local storages
    dscal(0.0, updateTerm.values)
    dscal(0.0, lossCache.values)

    // Evaluate gradient and loss at the current point
    addGradient(localWeights, features, labels, updateTerm, marginCache, lossCache)

    // Add L2 regularization to the loss and gradient
    if (l2regParam != null) {
      addL2Reg(l2regParam, localWeights, updateTerm, lossCache, skipRegFeature)
    }

    if (l1regParam != null) {
      addL1Reg(l1regParam, localWeights, updateTerm, lossCache, skipRegFeature)
    }
  }

  def axpyCompensated(updateTerm: Array[Double], sum: Array[Double], compensator: Array[Double], y: Array[Double], t: Array[Double]) = {
//    copy(updateTerm, y)
//    axpy(-1.0, compensator, y)
//
//    copy(sum, t)
//    axpy(1.0, y, t)
//
//    copy(t, compensator)
//    axpy(-1.0, sum, compensator)
//    axpy(-1.0, y, compensator)
//
//    copy(t, sum)

    axpy(1.0, updateTerm, sum)
  }

  /**
    * For single instance and weights calculates gradient and loss. Depending on direction adds gradient
    * and loss to the accumulated data.
    *
    * @param weights    Weights to evaluate gradient at
    * @param features   Featrues of instance to evaluate gradient at
    * @param labels     Labels of the instance to evaluate gradient at
    * @param updateTerm Update term to store gradient at
    * @param lossCache  Loss vector to record resulting loss values.
    */
  protected def addGradient(
                             weights: Matrix,
                             features: DenseMatrix,
                             labels: DenseMatrix,
                             updateTerm: DenseMatrix,
                             marginCache: DenseMatrix,
                             lossCache: DenseVector)

  /**
    * Adds L2 regularization part to the gradient and loss.
    */
  protected def addL2Reg(l2regParam: Vector, weights: DenseMatrix, updateTerm: DenseMatrix, lossCache: DenseVector, skipRegFeature: Int) = {
    MatrixUtils.applyNonZeros(
      weights,
      updateTerm,
      (label, feature, weight, grad) => {
        if (feature == skipRegFeature) {
          grad
        } else {
          lossCache.values(label) += 0.5 * l2regParam(label) * weight * weight
          grad + l2regParam(label) * weight
        }
      }
    )
  }


  protected def addL1Reg(l1regParam: Vector, weights: DenseMatrix, updateTerm: DenseMatrix, lossCache: DenseVector, skipRegFeature: Int) = {
    MatrixUtils.applyAll(
      weights,
      updateTerm,
      (label, feature, xv, v) => {
        if (feature == skipRegFeature) {
          v
        } else {
          val l1regValue = l1regParam(label)
          lossCache.values(label) += l1regValue * Math.abs(xv)

          // This part is taken from Breeze.OWLQN
          xv match {
            case 0.0 => {
              val delta_+ = v + l1regValue
              val delta_- = v - l1regValue
              if (delta_- > 0) delta_- else if (delta_+ < 0) delta_+ else 0.0
            }
            case _ => v + Math.signum(xv) * l1regValue
          }
        }
      }
    )
  }

  /**
    * Updates the weights given update term and current value.
    */
  protected def updateWeights(stepSize: Double, updateTerm: DenseMatrix, weights: DenseMatrix) = {
    //dscal(1 - stepSize, weights.values)
    axpy(-stepSize, updateTerm.values, weights.values)
  }

  /**
    * Apply L1 shrinkage to the updated weights.
    */
  protected def applyL1Shrinkage(regParam: Vector, weights: DenseMatrix, skipRegFeature: Int, notDegraded: Set[Int]) = {
    MatrixUtils.transformDense(
      weights,
      (label, feature, weight) => {
        if (feature == skipRegFeature || !notDegraded.contains(label)) {
          weight
        } else {
          val shrinkage = regParam(label)
          if (Math.abs(weight) > shrinkage) weight else 0.0
        }
      })
  }

  /**
    * Merges weights from the new epoch with overal weights. Dimensions of weights matrices might be
    * different when part of labels are already converged and do not participate in descend.
    */
  protected def merge(labelsMap: Map[Int,Int], weights: Matrix, newWeights: DenseMatrix): DenseMatrix = {
    if (weights.numRows == newWeights.numRows && weights.numRows == labelsMap.size) {
      newWeights
    } else {
      val result = weights match {
        case d: DenseMatrix => d
        case s: SparseMatrix => s.toDense
      }

      MatrixUtils.transformDense(
        result,
        (label, feature, weight) => labelsMap.get(label).map(x => newWeights(x, feature)).getOrElse(weight))
    }
  }

  /**
    * Extracts not converged labels based on actual and previous weights and on the loss history.
    */
  protected def getNotConverged(
                                 activeLabels: Map[Int,Int],
                                 lossHistory: Array[CompactBuffer[Double]],
                                 weightDiffHistory: Array[CompactBuffer[Double]],
                                 weightNormHistory: Array[CompactBuffer[Double]],
                                 tolerance: Double): Array[Int] = {
    activeLabels.keys.filterNot(label => {
      val weightsDistance: Double = weightDiffHistory(label).last

      val lossDelta = lossDifferenceForLabel(lossHistory, label)

      weightsDistance.isNaN || lossDelta.isNaN || ($(convergenceMode) match {
        case "weights" => weightsDistance < tolerance
        case "loss" => lossDelta < tolerance
        case "both" => weightsDistance < tolerance && lossDelta < tolerance
        case "any" => weightsDistance < tolerance || lossDelta < tolerance
      })
    }).toArray
  }

  /**
    * Evaluates loss difference simply as relative change
    */
  def lossDifferenceForLabel(lossHistory: Array[CompactBuffer[Double]], label: Int): Double = {
    val lastLosses = lossHistory(label).takeRight(2)
    val were = lastLosses.head
    val now = lastLosses.last

    Math.abs(were - now) / Math.max($(tol), Math.abs(were) + Math.abs(now))
  }

  /**
    * Evaluates weight distance based on old and new weights images.
    *
    * @param oldWeights Weights from the previous epoch
    * @param newWeights Weights from the current epoch.
    * @param label      Label to check for convergence.
    * @return Distance between old and new weights.
    */
  protected def weightsDistanceForLabel(oldWeights: Matrix, newWeights: DenseMatrix, label: Int): Double

  /**
    * Evaluates weight norm for a given label.
    *
    * @param newWeights Weights matrix
    * @param label      Label to evaluate weights
    * @return Weights norm.
    */
  protected def weightNorm(newWeights: Matrix, label: Int, skipRegFeature: Int): Double = {
    var sum = 0.0

    newWeights.foreachActive((i, feature, weight) => if (label == i && feature != skipRegFeature) sum += weight * weight)

    Math.sqrt(sum)
  }

  override def copy(extra: ParamMap): DSVRGD[M] =
    defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    val labelField: StructField = schema($(labelCol))
    SchemaUtils.appendColumn(
      schema,
      new StructField($(predictionCol), labelField.dataType, labelField.nullable, labelField.metadata))
  }

}

object DSVRGD extends Serializable with HasNetlibBlas with HasLossHistory {

  /**
    * Summary block used for keeping loss history.
    */
  val WeightDiffHistory = new Block("weightDiffHistory")
  val WeightNormHistory = new Block("weightNormHistory")

  /**
    * Helper used to store and mere epoch state
    */
  case class DistributedSgdState
  (weights: DenseMatrix,
   accumulatedGradient: DenseMatrix,
   accumulatedWeights: DenseMatrix,
   accumulatedLoss: DenseVector,
   partsCount: Int) extends HasNetlibBlas {

    def merge(other: DistributedSgdState): DistributedSgdState = {
      axpy(1.0, other.weights.values, weights.values)
      axpy(1.0, other.accumulatedGradient.values, accumulatedGradient.values)
      axpy(1.0, other.accumulatedWeights.values, accumulatedWeights.values)
      axpy(1.0, other.accumulatedLoss.values, accumulatedLoss.values)

      DistributedSgdState(weights, accumulatedGradient, accumulatedWeights, accumulatedLoss, partsCount + other.partsCount)
    }

    def scale(): DistributedSgdState = {
      val divider = 1.0 / partsCount

      dscal(divider, weights.values)
      dscal(divider, accumulatedGradient.values)
      dscal(divider, accumulatedWeights.values)
      dscal(divider, accumulatedLoss.values)

      this
    }

  }

  /**
    * Helper for creating loss history data frame
    */
  case class LossRecord(label: String, iteration: Int, loss: Double)

  /**
    * Linear regression gradient and loss (sguared loss)
    */
  def linear(
              weights: Matrix,
              features: DenseMatrix,
              labels: DenseMatrix,
              updateTerm: DenseMatrix,
              marginCache: DenseMatrix,
              lossCache: DenseVector) = {

    BLAS.gemm(1.0, weights, features, 0.0, marginCache)
    axpy(-1.0, labels.values, marginCache.values)

    val multiplier = 1.0 / features.numCols
    BLAS.gemm(
      multiplier,
      marginCache,
      features.transpose,
      0.0,
      updateTerm)

    marginCache.foreachActive((label, sample, v) => lossCache.values(label) += multiplier * v * v)
  }

  /**
    * Logistic regression gradiend and loss (logistic loss)
    */
  def logistic(
                weights: Matrix,
                features: DenseMatrix,
                labels: DenseMatrix,
                updateTerm: DenseMatrix,
                marginCache: DenseMatrix,
                lossCache: DenseVector) = {

    BLAS.gemm(-1.0, weights, features, 0.0, marginCache)

    val multiplier = 1.0 / features.numCols

    MatrixUtils.applyNonZeros(
      labels,
      marginCache,
      (label, sample, labelValue, margin) => {
        // loss_i += log(1 + exp(margin_i)) - (1 - label_i) * margin_i
        lossCache.values(label) += multiplier * (MLUtils.log1pExp(margin) - (1 - labelValue) * margin)

        // multiplier_i = 1 / (1 + exp(margin_i)) - label(i)
        1.0 / (1.0 + Math.exp(margin)) - labelValue
      }
    )

    BLAS.gemm(
      multiplier,
      marginCache,
      features.transpose,
      0.0,
      updateTerm)
  }

  def logisticInitialization(data: DataFrame, numLabels: Int, numFeatures: Int): Matrix = {
    val stat = data.rdd.map(_.getAs[Vector](0)).treeAggregate(
      new MultivariateOnlineSummarizer)(
      (a, v) => a.add(mllib.linalg.Vectors.fromML(v)), (a, b) => a.merge(b))

    MatrixUtils.transformDense(
      DenseMatrix.zeros(numLabels, numFeatures),
      (label, feature, weight) => {
        if (feature == numFeatures - 1) {
          /*
         For binary logistic regression, when we initialize the coefficients as zeros,
         it will converge faster if we initialize the intercept such that
         it follows the distribution of the labels.

         {{{
         P(0) = 1 / (1 + \exp(b)), and
         P(1) = \exp(b) / (1 + \exp(b))
         }}}, hence
         {{{
         b = \log{P(1) / P(0)} = \log{count_1 / count_0}
         }}}
       */
          Math.log(stat.numNonzeros(label) / stat.count)
        } else {
          0.0
        }
      })
  }

  /**
    * For linear regression weights are compared based on relative euclidean distance.
    */
  def linearWeightsDistance(oldWeights: Matrix, newWeights: DenseMatrix, label: Int): Double = {
    var diff = 0.0
    var sum = 0.0

    for (j <- 0 until newWeights.numCols) {
      sum += newWeights(label, j) * newWeights(label, j)
      diff += (oldWeights(label, j) - newWeights(label, j)) * (oldWeights(label, j) - newWeights(label, j))
    }

    Math.sqrt(diff) / Math.sqrt(sum)
  }

  /**
    * For logistic regression weights are compared based on cosine distance
    */
  def logisticWeightsDistance(oldWeights: Matrix, newWeights: DenseMatrix, label: Int): Double = {
    var cor = 0.0
    var sumNew = 0.0
    var sumOld = 0.0

    for (j <- 0 until newWeights.numCols) {
      sumNew += newWeights(label, j) * newWeights(label, j)
      sumOld += oldWeights(label, j) * oldWeights(label, j)
      cor += oldWeights(label, j) * newWeights(label, j)
    }

    if (sumNew * sumOld > 0) {
      1 - cor / Math.sqrt(sumNew * sumOld)
    } else {
      2
    }
  }
}

/**
  * Helper class for training single-label models.
  */
abstract class DeVectorizedDSVRGD[M <: ModelWithSummary[M]](override val uid: String)
  extends DSVRGD[M](uid) {

  override def fit(dataset: Dataset[_]): M = {
    val vectorize = functions.udf[Vector, Double](x => Vectors.dense(x))

    val labelField = dataset.schema($(labelCol))
    val attributes = new AttributeGroup($(labelCol), 1)
    val metadata = if (labelField.metadata != null) attributes.toMetadata(labelField.metadata) else attributes.toMetadata()

    super.fit(dataset.withColumn(
      $(labelCol),
      vectorize(
        dataset($(labelCol))).as($(labelCol),
        metadata)))
  }


  override def extractBlock(lossHistory: Array[CompactBuffer[Double]], dataset: DataFrame, names: Map[Int, String], sc: SparkContext): DataFrame =
    super.extractBlock(lossHistory, dataset, names, sc).drop($(labelCol))
}

/**
  * Multi-label linear regresion with DSVRGD
  */
class LinearMatrixDSVRGD(override val uid: String)
  extends DSVRGD[LinearCombinationModel[LinearRegressionModel]](uid) {

  def this() = this(Identifiable.randomUID("linearMatrixDSVRG"))

  protected override def addGradient(
                                      weights: Matrix,
                                      features: DenseMatrix,
                                      labels: DenseMatrix,
                                      updateTerm: DenseMatrix,
                                      marginCache: DenseMatrix,
                                      lossCache: DenseVector)
  = DSVRGD.linear(weights, features, labels, updateTerm, marginCache, lossCache)

  protected override def weightsDistanceForLabel(
                                                  oldWeights: Matrix,
                                                  newWeights: DenseMatrix,
                                                  label: Int): Double
  = DSVRGD.linearWeightsDistance(oldWeights, newWeights, label)


  protected override def extractModel
  (labelAttributeGroup: AttributeGroup, numLabels: Int, weights: Matrix, dataset: DataFrame): LinearCombinationModel[LinearRegressionModel] = {
    val result: Map[String, Vector] = extractLabelVectors(labelAttributeGroup, numLabels, weights)

    new LinearCombinationModel[LinearRegressionModel](result.map(
      x => x._1 -> LinearRegressionModel
        .create(x._2, dataset.sqlContext, dataset.schema.fields(dataset.schema.fieldIndex($(featuresCol))))
        .asInstanceOf[LinearRegressionModel]))
      .setPredictVector(result.keys.map(k => Map(k -> 1.0)).toSeq: _*)
  }
}

object LinearMatrixDSVRGD extends DefaultParamsReadable[LinearMatrixDSVRGD]

/**
  * Single-label linear regresion with DSVRGD
  */
class LinearDSVRGD(override val uid: String)
  extends DeVectorizedDSVRGD[LinearRegressionModel](uid) {

  def this() = this(Identifiable.randomUID("linearDSVRG"))

  protected override def addGradient(
                                      weights: Matrix,
                                      features: DenseMatrix,
                                      labels: DenseMatrix,
                                      updateTerm: DenseMatrix,
                                      marginCache: DenseMatrix,
                                      lossCache: DenseVector)
  = DSVRGD.linear(weights, features, labels, updateTerm, marginCache, lossCache)


  protected override def weightsDistanceForLabel(
                                                  oldWeights: Matrix,
                                                  newWeights: DenseMatrix,
                                                  label: Int): Double
  = DSVRGD.linearWeightsDistance(oldWeights, newWeights, label)


  protected override def extractModel
  (labelAttributeGroup: AttributeGroup, numLabels: Int, weights: Matrix, dataset: DataFrame): LinearRegressionModel = {
    val result: Map[String, Vector] = extractLabelVectors(labelAttributeGroup, numLabels, weights)

    LinearRegressionModel
      .create(result.values.head, dataset.sqlContext, dataset.schema.fields(dataset.schema.fieldIndex($(featuresCol))))
      .asInstanceOf[LinearRegressionModel]
  }
}

object LinearDSVRGD extends DefaultParamsReadable[LinearDSVRGD]

/**
  * Multi-label logistic regresion with DSVRGD
  */
class LogisticMatrixDSVRGD(override val uid: String)
  extends DSVRGD[LinearCombinationModel[LogisticRegressionModel]](uid) {

  def this() = this(Identifiable.randomUID("logisticMatrixDSVRG"))

  protected override def addGradient(
                                      weights: Matrix,
                                      features: DenseMatrix,
                                      labels: DenseMatrix,
                                      updateTerm: DenseMatrix,
                                      marginCache: DenseMatrix,
                                      lossCache: DenseVector)
  = DSVRGD.logistic(weights, features, labels, updateTerm, marginCache, lossCache)

  override def initializeWeights(data: DataFrame, numLabels: Int, numFeatures: Int): Matrix = {
    if ($(lastIsIntercept)) {
      DSVRGD.logisticInitialization(data.select($(labelCol)), numLabels, numFeatures)
    } else {
      super.initializeWeights(data, numLabels, numFeatures)
    }
  }

  protected override def weightsDistanceForLabel(
                                                  oldWeights: Matrix,
                                                  newWeights: DenseMatrix,
                                                  label: Int): Double
  = DSVRGD.logisticWeightsDistance(oldWeights, newWeights, label)

  // For logistic regression where is a scheme for evaluating maximal possible L1 regularisation
  // see http://jmlr.org/papers/volume8/koh07a/koh07a.pdf for details
  override protected def evaluateL1Regularization(data: DataFrame, l1Scalar: Double, numLabels: Int): Vector = {
    val regMax = MatrixLBFGS.evaluateMaxRegularization(data, $(featuresCol), $(labelCol), !$(lastIsIntercept))._2.toDense
    regMax.values.transform(_ * l1Scalar)
    regMax
  }

  protected override def extractModel
  (labelAttributeGroup: AttributeGroup, numLabels: Int, weights: Matrix, dataset: DataFrame): LinearCombinationModel[LogisticRegressionModel] = {
    val result: Map[String, Vector] = extractLabelVectors(labelAttributeGroup, numLabels, weights)

    new LinearCombinationModel[LogisticRegressionModel](result.map(
      x => x._1 -> LogisticRegressionModel
        .create(x._2, dataset.sqlContext, dataset.schema.fields(dataset.schema.fieldIndex($(featuresCol))))
        .asInstanceOf[LogisticRegressionModel]))
      .setPredictVector(result.keys.map(k => Map(k -> 1.0)).toSeq: _*)
  }
}

object LogisticMatrixDSVRGD extends DefaultParamsReadable[LogisticMatrixDSVRGD]

/**
  * Multi-label logistic regresion with DSVRGD
  */
class LogisticDSVRGD(override val uid: String)
  extends DeVectorizedDSVRGD[LogisticRegressionModel](uid) {

  def this() = this(Identifiable.randomUID("logisticDSVRG"))

  protected override def addGradient(
                                      weights: Matrix,
                                      features: DenseMatrix,
                                      labels: DenseMatrix,
                                      updateTerm: DenseMatrix,
                                      marginCache: DenseMatrix,
                                      lossCache: DenseVector)
  = DSVRGD.logistic(weights, features, labels, updateTerm, marginCache, lossCache)


  override def initializeWeights(data: DataFrame, numLabels: Int, numFeatures: Int): Matrix = {
    if ($(lastIsIntercept)) {
      DSVRGD.logisticInitialization(data.select($(labelCol)), numLabels, numFeatures)
    } else {
      super.initializeWeights(data, numLabels, numFeatures)
    }
  }

  protected override def weightsDistanceForLabel(
                                                  oldWeights: Matrix,
                                                  newWeights: DenseMatrix,
                                                  label: Int): Double
  = DSVRGD.logisticWeightsDistance(oldWeights, newWeights, label)

  // For logistic regression where is a scheme for evaluating maximal possible L1 regularisation
  // see http://jmlr.org/papers/volume8/koh07a/koh07a.pdf for details
  override protected def evaluateL1Regularization(data: DataFrame, l1Scalar: Double, numLabels: Int): Vector = {
    val regMax = MatrixLBFGS.evaluateMaxRegularization(data, $(featuresCol), $(labelCol), $(lastIsIntercept))._2.toDense
    regMax.values.transform(_ * l1Scalar)
    regMax
  }

  protected override def extractModel
  (labelAttributeGroup: AttributeGroup, numLabels: Int, weights: Matrix, dataset: DataFrame): LogisticRegressionModel = {
    val result: Map[String, Vector] = extractLabelVectors(labelAttributeGroup, numLabels, weights)

    LogisticRegressionModel
      .create(result.values.head, dataset.sqlContext, dataset.schema.fields(dataset.schema.fieldIndex($(featuresCol))))
      .asInstanceOf[LogisticRegressionModel]
  }
}

object LogisticDSVRGD extends DefaultParamsReadable[LogisticDSVRGD]


