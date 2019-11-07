package org.apache.spark.ml.odkl

import java.util.concurrent.atomic.{AtomicInteger, AtomicReferenceArray}
import java.util.concurrent.{ArrayBlockingQueue, CountDownLatch, ThreadPoolExecutor, TimeUnit}

import breeze.linalg
import breeze.optimize.{CachedDiffFunction, DiffFunction, OWLQN}
import odkl.analysis.spark.util.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasRegParam, HasTol}
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

import scala.collection.mutable
import scala.collection.parallel.ThreadPoolTaskSupport
import scala.collection.parallel.mutable.ParArray


/**
  * Created by dmitriybugaichenko on 24.03.16.
  *
  * Implementation for multi-class logistic regression training. In contrast to traditional notion of multi-class
  * logistic regression this trainer produces one regression per each class. Internally treats all classes
  * simultaneously using matrix-matrix multplication. Allows for L1-regularization (switches LBFGS to OWL-QN for
  * that). Regularization strength is defined in terms of fraction of maximal feasible regularization (deduced using
  * http://jmlr.org/papers/volume8/koh07a/koh07a.pdf).
  */
class MatrixLBFGS(override val uid: String) extends
  SummarizableEstimator[LinearCombinationModel[LogisticRegressionModel]]
  with PredictorParams with HasTol with HasMaxIter with HasRegParam with HasRegularizeLast with HasBatchSize
  with HasNetlibBlas {

  val predictVector = new Param[Boolean](this, "predictVector", "Whenever to configure model for predicting a vector.")
  val numCorrections = new  IntParam(this, "numCorrections", "Number of corrections to memorize for search in LBFGS.")

  def setRegParam(value: Double): this.type = set(regParam, value)

  def setPredictVector(value: Boolean): this.type = set(predictVector, value)

  setDefault(
    featuresCol -> "features",
    labelCol -> "label",
    tol -> 1E-4,
    maxIter -> 100,
    predictVector -> false,
    batchSize -> 200,
    numCorrections -> 10,
    regParam -> 0.0,
    regularizeLast -> true)

  def this() = this(Identifiable.randomUID("matrixLBFGS"))

  override def fit(dataset: Dataset[_]): LinearCombinationModel[LogisticRegressionModel] = {
    val result: Map[String, Vector] = MatrixLBFGS.multiClassLBFGS(
      dataset.toDF, $(featuresCol), $(labelCol), 10, $(tol), $(maxIter), $(batchSize), $(regParam), $(regularizeLast))

    val model = new LinearCombinationModel[LogisticRegressionModel](result.map(
      x => x._1 -> LogisticRegressionModel
        .create(x._2, dataset.sqlContext, dataset.schema.fields(dataset.schema.fieldIndex($(featuresCol))))
        .asInstanceOf[LogisticRegressionModel]))
      .setParent(this)

    if ($(predictVector)) {
      model.setPredictVector(result.keys.map(k => Map(k -> 1.0)).toSeq: _*)
    } else {
      model
    }
  }

  override def copy(extra: ParamMap): MatrixLBFGS = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    SchemaUtils.appendColumn(schema, $(predictionCol), new VectorUDT)
}

object MatrixLBFGS extends Logging with HasNetlibBlas {

  /**
    * Compute the gradient and loss given the features of a single data point.
    *
    * @param data    features for one data point
    * @param label   label for this data point
    * @param weights weights/coefficients corresponding to features
    * @return Loss vector for the current point.
    */
  def computeGradient
  (
    data: Vector,
    label: Vector,
    weights: DenseMatrix,
    accumulatedGradient: DenseMatrix,
    accumulatedLoss: DenseVector)
  : Unit = {
    computeGradientMatrix(
      data.toArray,
      label.toArray,
      weights,
      accumulatedGradient,
      accumulatedLoss,
      new Array[Double](label.size),
      1)
  }

  /**
    * Computes gradient and loss for a batch containing data and labels from multiple samples.
    *
    * @param data                Samples matrix in row-major form (one row per sample)
    * @param label               Labels matrix in row-major form (one row per sample)
    * @param weights             Matrix with weights (column-major)
    * @param accumulatedGradient Matrix with accumulated gradient
    * @param accumulatedLoss     Vector with accumulated loss
    * @param marginCache         Array used to cache margin calculations
    * @param samples             Number of samples in the batch
    */
  def computeGradientMatrix
  (
    data: Array[Double],
    label: Array[Double],
    weights: DenseMatrix,
    accumulatedGradient: DenseMatrix,
    accumulatedLoss: DenseVector,
    marginCache: Array[Double],
    samples: Int)
  : Unit = {

    val numLabels = weights.numRows
    val numFeatures = weights.numCols

    // margin = -W * x
    gemm(
      alpha = -1.0,
      beta = 0.0,
      A = weights.values,
      B = data,
      C = marginCache,
      aNumRows = weights.numRows,
      aNumCols = weights.numCols,
      // Note that for blas the native form is column-major, thus we here are pretending that data is a column-major
      // matrix with samples as a column
      bNumRows = numFeatures,
      bNumCols = samples,
      aIsTransposed = weights.isTransposed,
      bIsTransposed = false)


    for (i <- 0 until numLabels * samples) {
      // loss_i += log(1 + exp(margin_i)) - (1 - label_i) * margin_i
      // multiplier_i = 1 / (1 + exp(margin_i)) - label(i)
      val labelIndex: Int = i % numLabels
      accumulatedLoss.values(labelIndex) += MLUtils.log1pExp(marginCache(i)) - (1 - label(i)) * marginCache(i)
      marginCache(i) = 1.0 / (1.0 + Math.exp(marginCache(i))) - label(i)
    }

    // grad_i += multiplier_i * x
    gemm(
      alpha = 1.0,
      beta = 1.0,
      A = marginCache,
      B = data,
      C = accumulatedGradient.values,
      aNumRows = numLabels,
      aNumCols = samples,
      bNumRows = samples,
      bNumCols = numFeatures,
      aIsTransposed = false,
      // And here we are telling the truth that - this is a transposed matrix (row-major, not column-major)
      bIsTransposed = true)
  }

  /**
    * Computes cumulative gradient and loss for a set of samples
    *
    * @param data           RDD with vectors of features and vectors of labels
    * @param currentWeights Current weights matrix.
    * @param batchSize      Number of samples to collect in a batch before computing.
    * @param labelsAssigner Routine for extracting labels vector (in some cases only part of the labels are needed)
    * @return Tuple with cumulative gradient matrix and loss vector
    */
  def computeGradientAndLoss[T]
  (
    data: RDD[(Vector, T)],
    currentWeights: DenseMatrix,
    batchSize: Int = 10,
    labelsAssigner: (Int, T, Array[Double]) => Unit)
  : (DenseMatrix, DenseVector) = {

    val broadcastWeights = data.sparkContext.broadcast(currentWeights)
    val numLabels = currentWeights.numRows
    val numFeatures = currentWeights.numCols

    try {
      val partitions: RDD[(DenseMatrix, DenseVector)] = data
        .mapPartitions(iter => {
          val grad = DenseMatrix.zeros(currentWeights.numRows, currentWeights.numCols)
          val loss = Vectors.zeros(currentWeights.numRows).toDense

          val data = new Array[Double](numFeatures * batchSize)
          val labels = new Array[Double](numLabels * batchSize)
          val marginCache = new Array[Double](numLabels * batchSize)
          var batched = 0

          iter.foreach(r => {
            System.arraycopy(r._1.toArray, 0, data, batched * numFeatures, numFeatures)
            labelsAssigner(batched * numLabels, r._2, labels)

            batched += 1
            if (batched >= batchSize) {
              computeGradientMatrix(
                data,
                labels,
                broadcastWeights.value,
                grad,
                loss,
                marginCache,
                batched
              )
              batched = 0
            }
          })

          if (batched > 0) {
            computeGradientMatrix(
              data,
              labels,
              broadcastWeights.value,
              grad,
              loss,
              marginCache,
              batched
            )
          }

          Iterator(grad -> loss)
        })

      partitions
        .treeReduce((first, second) => {
          axpy(1.0, second._1.values, first._1.values)
          axpy(1.0, second._2.values, first._2.values)

          first
        })
    } finally {
      broadcastWeights.destroy()
    }
  }

  /**
    * Implementation of the matrix LBFGS algorithm. Uses breeze implementation of the iterations and
    * provides it with a specific cost function. The function batches requests for costs for different
    * labels and converts to a single matrix pass.
    *
    * @param data             Data fram to run on.
    * @param featuresColumn   Name of the column with features vector. Attribute group metadata is required
    * @param labelColumn      Name of the column with labels vector. Attribute group metadata is required
    * @param numCorrections   Number of corrections in LBFGS iteration
    * @param convergenceTol   Convergence tolerance for the iteration
    * @param maxNumIterations Maximum number of iteration
    * @param batchSize        Number of samples to batch before calculating
    * @return Map label -> trained weights vector
    */
  def multiClassLBFGS
  (
    data: DataFrame,
    featuresColumn: String,
    labelColumn: String,
    numCorrections: Int,
    convergenceTol: Double,
    maxNumIterations: Int,
    batchSize: Int,
    regParam: Double = 0.0,
    regulaizeLast: Boolean = true): Map[String, Vector] = {

    val labels = AttributeGroup.fromStructField(data.schema(labelColumn))
    val features = AttributeGroup.fromStructField(data.schema(featuresColumn))

    val (numElements, regMax) = if (regParam > 0) {
      evaluateMaxRegularization(
        data, featuresColumn, labelColumn, regulaizeLast)
    } else {
      (data.count(), Vectors.zeros(labels.size))
    }

    logInfo(s"Deduced max regularization settings $regMax for labels $labels")

    val batchCostFunction = new BatchCostFunction(
      data.select(featuresColumn, labelColumn).rdd.map(r => (r.getAs[Vector](0), r.getAs[Vector](1))),
      features.size,
      labels.size,
      numElements,
      batchSize
    )


    val attributes: Array[Attribute] = labels.attributes
      .getOrElse(Array.tabulate(labels.size) { i => NumericAttribute.defaultAttr.withIndex(i) })

    batchCostFunction.reset()

    val initializer: ParArray[Attribute] = attributes.par
    val support = new ThreadPoolTaskSupport(
      new ThreadPoolExecutor(labels.size, labels.size, Long.MaxValue, TimeUnit.DAYS, new ArrayBlockingQueue[Runnable](labels.size)))
    support.environment.prestartAllCoreThreads()

    initializer.tasksupport = support

    logInfo("Initializing LBFGS states...")
    val allProcessors = initializer.map(a => new LbfgsState(
      a.index.get,
      numCorrections,
      convergenceTol,
      maxNumIterations,
      Vectors.zeros(features.size),
      batchCostFunction,
      regParam * regMax(a.index.get),
      regulaizeLast
    )).toArray
    logInfo("Initialization done.")

    var iteration = 1
    val active = new AtomicInteger(0)

    do {
      logInfo(s"Starting iteration $iteration")
      batchCostFunction.reset()
      active.set(0)
      val countDown = new CountDownLatch(labels.size)

      allProcessors.foreach(processor =>
        support.environment.execute(new Runnable {
          override def run(): Unit = try {
            logInfo(s"Dealing with ${processor.label} at ${Thread.currentThread().getName}")
            if (processor.hasNext) {
              logInfo(s"Executing with ${processor.label} at ${Thread.currentThread().getName}")
              processor.next()
              active.incrementAndGet()
            }
          }
          finally {
            logInfo(s"Done with ${processor.label} at ${Thread.currentThread().getName}")
            batchCostFunction.doneIter(processor.label)
            countDown.countDown()
          }
        })
      )

      countDown.await()
      logInfo(s"Processors still active ${active.get()} at iteration $iteration")
      iteration += 1
    } while (active.get() > 0)

    support.environment.shutdown()

    allProcessors
      .map(p => attributes(p.label).name.getOrElse(p.label.toString) -> Vectors.fromBreeze(p.state.x))
      .toMap
  }

  /**
    * Evaluates upper bound for regularization param for each label based on estimation from
    * http://jmlr.org/papers/volume8/koh07a/koh07a.pdf
    *
    * @param data Dataframewith samples.
    * @param featuresColumn Name of the features column
    * @param labelColumn Name of the labels column.
    * @param regulaizeLast Whenever to consider last feature as a subject for regularization (set to false
    *                      to exclude intercept from regularization)
    * @return A pair with instances count and regularization bounds.
    */
  def evaluateMaxRegularization
  (
    data: DataFrame,
    featuresColumn: String,
    labelColumn: String,
    regulaizeLast: Boolean) : (Long, Vector) = {


    val labels = AttributeGroup.fromStructField(data.schema(labelColumn))
    val features = AttributeGroup.fromStructField(data.schema(featuresColumn))

    val rdd = data.toDF.select(featuresColumn, labelColumn)
        .rdd.map(x => x.getAs[Vector](0) -> x.getAs[Vector](1))

    val labelsStat = rdd.map(_._2).mapPartitions(i => {
      val aggregator = new MultivariateOnlineSummarizer()

      i.foreach(x => aggregator.add(mllib.linalg.Vectors.fromML(x)))

      Iterator(aggregator)
    }).treeReduce(_ merge _)

    (labelsStat.count, evaluateMaxRegularization(
      rdd,
      regulaizeLast, features.size, labelsStat.mean.asML.toDense, labelsStat.count))
  }

  /**
    * Evaluates upper bound for regularization param for each label based on estimation from
    * http://jmlr.org/papers/volume8/koh07a/koh07a.pdf
    *
    * @param data RDD with samples features -> labels.
    * @param regulaizeLast Whenever to consider last feature as a subject for regularization (set to false
    *                      to exclude intercept from regularization)
    * @return A pair with instances count and regularization bounds.
    */
  def evaluateMaxRegularization
  (
    data: RDD[(Vector,Vector)],
    regulaizeLast: Boolean,
    numFeatures: Int,
    labelsMean: DenseVector,
    numExamples: Long): Vector = {

    val numLabels = labelsMean.size
    val multiplier = 1.0 / numExamples

    val correlations = data.mapPartitions(i => {
      val result = DenseMatrix.zeros(numFeatures, numLabels)

      for (row <- i) {
        val features = row._1
        val labels = row._2.toDense

        axpy(-1.0, labelsMean.toArray, labels.toArray)

        // Single initial pass, does not worth batching.
        gemm(
          multiplier,
          1.0,
          features.toArray,
          labels.toArray,
          result.values,
          features.size,
          1,
          1,
          labels.size,
          aIsTransposed = false,
          bIsTransposed = false
        )
      }

      Iterator(result)
    }).treeReduce((a, b) => {
      axpy(1.0, a.values, b.values)
      b
    })

    val bound = if (regulaizeLast) numFeatures else numFeatures - 1
    val regMax = if(bound > 0) Array.tabulate(numLabels) { j => (0 until bound).map(i => Math.abs(correlations(i, j))).max } else Array.fill(numLabels)(0.0)

    Vectors.dense(regMax)
  }

  /**
    * A bit messy class used to integrate LBFGS with matrix gradient calculation. The idea is to run multiple
    * LBFGS instances per each label and provide it with a dedicated vector cost function. Then accumulate all
    * calls from LBFGS, start calculation on a combined matrix and construct multiple results per each function.
    * Note that it is expected that LBFGSs are executed in parallel.
    */
  private class BatchCostFunction
  (
    data: RDD[(Vector, Vector)],
    numFeatures: Int,
    numLabels: Int,
    numItems: Long,
    batchSize: Int) extends Logging {

    val zeros: linalg.DenseVector[Double] = linalg.DenseVector.zeros(numFeatures)

    def doneIter(index: Int) = signal.synchronized {
      vectors.set(index, null)
      logInfo(s"Marking done for index $index")
      if (waiting.get() + ready.incrementAndGet() == numLabels && waiting.get() > 0) {
        logInfo(s"Calculating from done for index $index")
        calculateLoss()
      }
    }

    private val vectors = new AtomicReferenceArray[linalg.DenseVector[Double]](numLabels)
    @volatile private var result: Map[Int, (linalg.DenseVector[Double], Double)] = Map()
    private val waiting = new AtomicInteger(0)
    private var ready = new AtomicInteger(0)
    private val signal = new Object()

    def reset() = {
      signal.synchronized {
        waiting.set(0)
        ready.set(0)
        for (i <- 0 until numLabels) vectors.set(i, null)
      }
    }

    def calculate(index: Int)(x: linalg.DenseVector[Double]): (Double, linalg.DenseVector[Double]) = {
      signal.synchronized {
        vectors.set(index, x)

        try {
          if (waiting.incrementAndGet() + ready.get() == numLabels) {
            logInfo(s"Calculating from calculate for index $index")
            calculateLoss()
          } else {
            logInfo(s"Not yet ready for $index, waiting")
            logInfo(s"Waiting $waiting, ready $ready")
            signal.wait()
          }

          logInfo(s"Constructing result for $index")
          val grad: linalg.DenseVector[Double] = result(index)._1
          val loss: Double = result(index)._2 / numItems
          (loss, grad)
        } finally {
          waiting.decrementAndGet()
        }
      }
    }

    def calculateLoss(): Unit = {
      val activeLabels =
        (for (i <- 0 until vectors.length()) yield i -> vectors.get(i))
          .filterNot(_._2 == null)
          .toArray

      val indices: Array[(Int, Int)] = activeLabels.map(_._1).zipWithIndex

      val numActiveLabels: Int = activeLabels.length

      val weights = new DenseMatrix(
        numActiveLabels,
        numFeatures,
        Array.tabulate(numActiveLabels * numFeatures) {
          i => activeLabels(i % numActiveLabels)._2(i / numActiveLabels)
        },
        isTransposed = false)


      val computed = if (numLabels == numActiveLabels) {
        computeGradientAndLoss[Vector](data, weights, batchSize,
          labelsAssigner = (pos, vector, target) => System.arraycopy(vector.toArray, 0, target, pos, vector.size))
      } else {

        computeGradientAndLoss[Vector](
          data, weights, batchSize,
          labelsAssigner = (pos, vector, array) => indices.foreach(i => array(i._2 + pos) = vector(i._1))
        )
      }

      dscal(1.0 / numItems, computed._1.values)

      result = activeLabels.map(x => x._1 -> {
        val id = indices.find(_._1 == x._1).get

        val localLoss = computed._2(id._2)
        val localGrad = vectorForRow(id._2, computed._1)

        (localGrad, localLoss)
      }).toMap

      logInfo("Calculation done, signal everyone")
      signal.notifyAll()
    }

    def vectorForRow(index: Int, matrix: DenseMatrix): linalg.DenseVector[Double] = {
      linalg.DenseVector.tabulate(numFeatures) { i => matrix(index, i) }
    }
  }

  /**
    * Captures a state of a breeze iteration process, backed by a batched cost function.
    */
  private class LbfgsState
  (
    val label: Int,
    numCorrections: Int,
    convergenceTol: Double,
    maxNumIterations: Int,
    initials: Vector,
    batchCost: BatchCostFunction,
    regParam: Double,
    regulaizeLast: Boolean) extends DiffFunction[breeze.linalg.DenseVector[Double]] {

    val lbfgs = if (regParam > 0.0) {
      new OWLQN[
        Int,
        breeze.linalg.DenseVector[Double]](
        maxNumIterations, numCorrections, (i: Int) => if (regulaizeLast || i < initials.size - 1) regParam else 0.0, convergenceTol)
    } else {
      new breeze.optimize.LBFGS[breeze.linalg.DenseVector[Double]](
        maxNumIterations, numCorrections, convergenceTol)
    }


    val iterations: Iterator[lbfgs.State] = lbfgs.iterations(
      new CachedDiffFunction(this), initials.asBreeze.toDenseVector)

    val lossHistory = mutable.ArrayBuilder.make[Double]

    def hasNext: Boolean = iterations.hasNext

    var state: lbfgs.State = null

    def next() = {
      state = iterations.next()
      lossHistory += state.value
    }

    override def calculate(x: breeze.linalg.DenseVector[Double]): (Double, breeze.linalg.DenseVector[Double]) =
      batchCost.calculate(label)(x)
  }


  /**
    * C := alpha * A * B + beta * C
    * For `DenseMatrix` A.
    */
  private def gemm(alpha: Double, beta: Double
                   , A: Array[Double]
                   , B: Array[Double]
                   , C: Array[Double]
                   , aNumRows: Int
                   , aNumCols: Int
                   , bNumRows: Int
                   , bNumCols: Int
                   , aIsTransposed: Boolean
                   , bIsTransposed: Boolean): Unit = {
    val tAstr = if (aIsTransposed) "T" else "N"
    val tBstr = if (bIsTransposed) "T" else "N"
    val lda = if (!aIsTransposed) aNumRows else aNumCols
    val ldb = if (!bIsTransposed) bNumRows else bNumCols

    blas.dgemm(tAstr, tBstr, aNumRows, bNumCols, aNumCols, alpha, A, lda,
      B, ldb, beta, C, aNumRows)
  }
}
