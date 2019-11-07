package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains classes used for linear models training with default implementations.
  */

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, OWLQN, LBFGS => BreezeLBFGS}
import odkl.analysis.spark.util.Logging
import org.apache.spark.annotation.Since
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.odkl.CombinedModel.DirectPredictionModel
import org.apache.spark.ml.odkl.ModelWithSummary.WithSummaryReader
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Predictor, PredictorParams}
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib
import org.apache.spark.mllib.optimization._
import org.apache.spark.rdd.RDD
import org.apache.spark.repro.ReproContext
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}

import scala.collection.mutable

trait LinearModelParams extends PredictorParams

abstract class LinearModel[M <: LinearModel[M]] private[odkl](override val uid: String)
  extends DirectPredictionModel[Vector, M] with ModelWithSummary[M] with LinearModelParams with HasWeights {

  def this() = this(Identifiable.randomUID("linearModel"))

  final val coefficients: Param[Vector] = new Param[Vector](this, "coefficients", "Weight coefficients for the model.")
  final val intercept: DoubleParam = new DoubleParam(this, "intercept", "Intercept for the model")

  setDefault(intercept, 0.0)
  setDefault(coefficients, Vectors.dense(Array[Double]()))

  def getCoefficients: Vector = $(coefficients)

  def getIntercept: Double = $(intercept)

  override final def predict(features: Vector): Double =
    postProcess(BLAS.dot($(coefficients), features) + $(intercept))

  def postProcess(value: Double): Double

  protected def init(coefficients: Vector, sqlContext: SQLContext, features: StructField) = {

    val attributes: AttributeGroup = AttributeGroup.fromStructField(features)

    val names = attributes.attributes.map(x => {
      x.map(a => a.index.get -> a.name.orNull).toMap
    }).getOrElse(Map())

    val summary =
      coefficients.toArray.zipWithIndex.map(x => WeightedFeature(x._2, names.get(x._2).orNull, x._1))

    val weightsBlock = SparkSqlUtils.reflectionLock.synchronized(sqlContext.createDataFrame(
      sqlContext.sparkContext.parallelize(summary, 1)))

    set(this.coefficients, coefficients)
    set(summaryParam, new ModelSummary(Map(weights -> weightsBlock)))
  }
}

class LinearRegressionModel private[odkl](uid: String = Identifiable.randomUID("linear"))
  extends LinearModel[LinearRegressionModel](uid) {

  override def create(): LinearRegressionModel = new LinearRegressionModel()

  override def postProcess(value: Double): Double = value
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  @Since("1.6.0")
  override def read: MLReader[LinearRegressionModel] = new WithSummaryReader[LinearRegressionModel]

  def create(coefficients: Vector, sqlContext: SQLContext, features: StructField) =
    new LinearRegressionModel().init(coefficients, sqlContext, features)
}

class LogisticRegressionModel private[odkl](uid: String = Identifiable.randomUID("logistic"))
  extends LinearModel[LogisticRegressionModel](uid) {

  override def create(): LogisticRegressionModel = new LogisticRegressionModel()

  override def postProcess(value: Double): Double = 1.0 / (1.0 + Math.exp(-value))
}

object LogisticRegressionModel extends MLReadable[LogisticRegressionModel] {
  @Since("1.6.0")
  override def read: MLReader[LogisticRegressionModel] = new WithSummaryReader[LogisticRegressionModel]

  def create(coefficients: Vector, sqlContext: SQLContext, features: StructField) =
    new LogisticRegressionModel().init(coefficients, sqlContext, features)
}

abstract class LinearEstimator[
M <: LinearModel[M],
T <: LinearEstimator[M, T]]
  extends Predictor[Vector, T, M] with SummarizableEstimator[M] with LinearModelParams with HasWeights {

  override def copy(extra: ParamMap): T
}

abstract class LinearRegressor[M <: LinearModel[M], O <: Optimizer, T <: LinearRegressor[M, O, T]]
(
  override val uid: String
)
  extends LinearEstimator[M, T] with DefaultParamsWritable with HasCacheTrainData {

  setDefault(cacheTrainData, true)

  override def copy(extra: ParamMap): T = defaultCopy(extra)

  protected override def train(dataset: Dataset[_]): M = {

    val data: RDD[(Double, mllib.linalg.Vector)] = dataset.select($(labelCol), $(featuresCol))
        .rdd.map(r => (r.getAs[Number](0).doubleValue(), mllib.linalg.Vectors.fromML(r.getAs[Vector](1))))

    val operationalData = if ($(cacheTrainData)) {
      data.cache()
    } else data

    try {

      val features = dataset.schema.fields(dataset.schema.fieldIndex($(featuresCol)))
      val attributes = AttributeGroup.fromStructField(features)

      val numFeatures = if (attributes.size > 0) {
        attributes.size
      } else {
        operationalData.take(1).head._2.size
      }

      //val initials = Vectors.zeros(numFeatures)
      val initials: Vector = SignificantFeatureSelector.tryGetInitials(features).getOrElse(Vectors.zeros(numFeatures))

      require(numFeatures == initials.size, "Got different sizes for initial weights and numFeatures")

      val optimizer: O = createOptimizer()
      val coefficients: mllib.linalg.Vector = optimizer.optimize(operationalData, mllib.linalg.Vectors.fromML(initials))

      createModel(optimizer, coefficients.asML, dataset.sqlContext, features)

    } finally {
      if ($(cacheTrainData)) {
        operationalData.unpersist()
      }
    }
  }

  protected def createModel(optimizer: O, coefficients: Vector, sqlContext: SQLContext, features: StructField): M

  protected def createOptimizer(): O

  protected def createWeightsSummary(coefficients: Vector, sqlContext: SQLContext, features: StructField): DataFrame = {

    val attributes: AttributeGroup = AttributeGroup.fromStructField(features)

    val names = attributes.attributes.map(x => {
      x.map(a => a.index.get -> a.name.orNull).toMap
    }).getOrElse(Map())

    val summary =
      coefficients.toArray.zipWithIndex.map(x => WeightedFeature(x._2, names.get(x._2).orNull, x._1))

    SparkSqlUtils.reflectionLock.synchronized(sqlContext.createDataFrame(
      sqlContext.sparkContext.parallelize(summary, 1)))
  }
}


class LogisticRegressionLBFSG(override val uid: String)
  extends LinearRegressor[LogisticRegressionModel, LogisticRegressionLBFSG, LogisticRegressionLBFSG](uid)
    with HasRegParam with HasTol with HasMaxIter with Optimizer with HasElasticNetParam with HasRegularizeLast
    with HasBatchSize {

  def this() = this(Identifiable.randomUID("logisticLBFSG"))

  def setRegParam(value: Double) : this.type = set(regParam, value)
  def setElasticNetParam(value: Double) : this.type = set(elasticNetParam, value)
  def setPreInitIntercept(value: Boolean): this.type = set(preInitIntercept, value)
  def setTol(value: Double): this.type = set(tol, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  final val numCorrections: IntParam = new IntParam(this, "numCorrections", "the number of corrections used in the LBFGS update." +
    "Values of numCorrections less than 3 are not recommended; large values of numCorrections will result in excessive computing time." +
    "3 < numCorrections < 10 is recommended.", ParamValidators.gt(0))

  final val preInitIntercept : BooleanParam = new BooleanParam(
    this, "preInitIntercept", "Whenever to pre-init intercept following labels distribution as b = log{P(1) / P(0)} = log{count_1 / count_0}")

  final def getNumCorrections: Int = $(numCorrections)

  setDefault(tol, 1E-4)
  setDefault(regParam, 0.0)
  setDefault(maxIter, 100)
  setDefault(numCorrections, 10)
  setDefault(elasticNetParam, 0.0)
  setDefault(regularizeLast, true)
  setDefault(batchSize, 200)
  setDefault(preInitIntercept, false)

  override protected def createModel(optimizer: LogisticRegressionLBFSG, coefficients: Vector, sqlContext: SQLContext, features: StructField): LogisticRegressionModel =
    LogisticRegressionModel.create(coefficients, sqlContext, features).setParent(this)

  override protected def createOptimizer(): LogisticRegressionLBFSG = this.copy(ParamMap())

  override def optimize(data: RDD[(Double, mllib.linalg.Vector)], initialWeights: mllib.linalg.Vector): mllib.linalg.Vector = {
    val lossHistory = mutable.ArrayBuilder.make[Double]

    val (sumLabel, numExamples) = data.aggregate(0.0 -> 0L)(
      (pair: (Double, Long), item: (Double, mllib.linalg.Vector)) => (pair._1 + item._1) -> (pair._2 + 1),
      (a: (Double, Long), b: (Double, Long)) => (a._1 + b._1) -> (a._2 + b._2)
    )

    val regParamL2 = $(regParam) * (1 - $(elasticNetParam))
    val regParamL1 = $(regParam) * $(elasticNetParam)

    val costFun =
      new CostFun(data, regParamL2, numExamples, $(regularizeLast), $(batchSize))

    val effectiveInitials = if ($(preInitIntercept)) {
      val result = initialWeights.toDense
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
      result.values(result.size - 1) = Math.log(sumLabel / (numExamples - sumLabel))
      result
    } else {
      initialWeights
    }

    val lbfgs = if (regParamL1 > 0.0 ) {
      // For logistic regression where is a scheme for evaluating maximal possible L1 regularisation
      // see http://jmlr.org/papers/volume8/koh07a/koh07a.pdf for details
      val effectiveRegL1 = MatrixLBFGS.evaluateMaxRegularization(
        data.map(x => x._2.asML -> Vectors.dense(x._1)),
        $(regularizeLast), numFeatures =  initialWeights.size,
        labelsMean = Vectors.dense(sumLabel / numExamples).toDense,
        numExamples
      )(0) * regParamL1

      val selector = if ($(regularizeLast)) {
        (_: Int) => effectiveRegL1
      } else {
        (i: Int) => if (i < initialWeights.size - 1) effectiveRegL1 else 0.0
      }

      new OWLQN[
        Int,
        BDV[Double]](
        $(maxIter), $(numCorrections),
        selector,
        $(tol))
    } else {
      new BreezeLBFGS[BDV[Double]]($(maxIter), $(numCorrections), $(tol))
    }

    val states =
      lbfgs.iterations(new CachedDiffFunction(costFun), effectiveInitials.asBreeze.toDenseVector)

    /**
      * NOTE: lossSum and loss is computed using the weights from the previous iteration
      * and regVal is the regularization value computed in the previous iteration as well.
      */
    var state = states.next()
    while (states.hasNext) {
      lossHistory += state.value
      state = states.next()
    }
    lossHistory += state.value
    val weights = Vectors.fromBreeze(state.x)

    val lossHistoryArray = lossHistory.result()

    logInfo(s"LBFGS.runLBFGS finished in ${lossHistoryArray.length} iterations. Last 10 " +
      s"losses ${lossHistoryArray.takeRight(10).mkString(", ")}")

    mllib.linalg.Vectors.fromML(weights)
  }

  case class LossInfo(name: String, value: Double)

  /**
    * CostFun implements Breeze's DiffFunction[T], which returns the loss and gradient
    * at a particular point (weights). It's used in Breeze's convex optimization routines.
    */
  private class CostFun(
                         data: RDD[(Double, mllib.linalg.Vector)],
                         regParamL2: Double,
                         numExamples: Long,
                         regularizeLast: Boolean,
                         batchSize: Int) extends DiffFunction[BDV[Double]] {

    override def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {


      val weightsArray = weights.toArray
      val result: (DenseMatrix, DenseVector) = MatrixLBFGS.computeGradientAndLoss[Double](
        data.map(x => x._2.asML -> x._1),
        Matrices.dense(1, weights.size, weightsArray).asInstanceOf[DenseMatrix],
        batchSize,
        (pos, label, target) => target(pos) = label
      )

      val discount = 1.0 / numExamples
      val grad: Array[Double] = result._1.values
      grad.transform(_ * discount)
      val loss = result._2(0) * discount

      if (regParamL2 == 0) {
        (loss, BDV(grad))
      } else {
        var regValue = 0.0

        for(i <- weightsArray.indices) {
          if (regularizeLast || i + 1 < weightsArray.length) {
            val weight = weightsArray(i)
            regValue = regValue + weight * weight
            grad(i) = grad(i) + regParamL2 * weight
          }
        }


        val regLoss = regValue * regParamL2 * 0.5
        (loss + regLoss, BDV(grad))
      }
    }
  }

}

class LinearRegressionSGD(override val uid: String)
  extends LinearRegressor[LinearRegressionModel, GradientDescent, LinearRegressionSGD](uid)
  with HasRegParam with HasTol with HasMaxIter with HasStepSize {

  def this() = this(Identifiable.randomUID("linearSGD"))

  def setRegParam(value: Double) : this.type = set(regParam, value)

  final val miniBatchFraction: DoubleParam = new DoubleParam(this, "miniBatchFraction", "fraction of data to be used for each SGD iteration.")

  final def getMiniBatchFraction: Double = $(miniBatchFraction)

  setDefault(tol, 0.001)
  setDefault(regParam, 0.0)
  setDefault(maxIter, 100)
  setDefault(stepSize, 1.0)
  setDefault(miniBatchFraction, 1.0)

  override protected def createModel(optimizer: GradientDescent, coefficients: Vector, sqlContext: SQLContext, features: StructField): LinearRegressionModel =
    LinearRegressionModel.create(coefficients, sqlContext, features).setParent(this)

  override protected def createOptimizer(): GradientDescent = new GradientDescent(new LeastSquaresGradient(), new SimpleUpdater())
    .setConvergenceTol($(tol))
    .setRegParam($(regParam))
    .setNumIterations($(maxIter))
    .setStepSize($(stepSize))
    .setMiniBatchFraction($(miniBatchFraction))
}

object LogisticRegressionLBFSG extends DefaultParamsReadable[LogisticRegressionLBFSG]

object LinearRegressionSGD extends DefaultParamsReadable[LinearRegressionSGD]





