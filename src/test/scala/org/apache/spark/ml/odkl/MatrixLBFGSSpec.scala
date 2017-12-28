package org.apache.spark.ml.odkl

import java.util.concurrent.ThreadLocalRandom

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, UserDefinedFunction, functions}
import org.scalatest.FlatSpec


/**
  * Created by vyacheslav.baranov on 02/12/15.
  */
class MatrixLBFGSSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations with WithTestData {

  val hiddenModel2 = Vectors.dense(0.7, -0.1)
  val hiddenModel3 = Vectors.dense(-0.3, 0.4)


  private val calculateLabels: UserDefinedFunction = functions.udf[Vector, Vector](x => Vectors.dense(
    WithTestData.logit(BLAS.dot(x, hiddenModel)),
    WithTestData.logit(BLAS.dot(x, hiddenModel2)),
    WithTestData.logit(BLAS.dot(x, hiddenModel3))
  ))

  lazy val multiClassData = noInterceptData.withColumn(
    "label", calculateLabels(noInterceptData("features")).as("label", new AttributeGroup("label", 3).toMetadata()))

  lazy val weights = Matrices.dense(3, 2, Array(
    hiddenModel(0), hiddenModel2(0), hiddenModel3(0),
    hiddenModel(1), hiddenModel2(1), hiddenModel3(1)
  )).asInstanceOf[DenseMatrix]

  lazy val multiclassRdd: RDD[(Vector, Vector)] = multiClassData.select("features", "label").map(r => r.getAs[Vector](0) -> r.getAs[Vector](1))

  lazy val matrixResult = MatrixLBFGS.computeGradientAndLoss[Vector](multiclassRdd, weights, batchSize = 13,
    labelsAssigner = (pos, vector, target) => System.arraycopy(vector.toArray, 0, target, pos, vector.size))

  lazy val referenceGradient = new LogisticGradient()

  lazy val reference1: (Vector, Double) = reference(multiClassData, 0, hiddenModel)

  lazy val reference2: (Vector, Double) = reference(multiClassData, 1, hiddenModel2)

  lazy val reference3: (Vector, Double) = reference(multiClassData, 2, hiddenModel3)

  def reference(data: DataFrame, label: Int, model: Vector) = {
    val referenceGradient = new LogisticGradient()
    data
      .select("label", "features")
      .map(r => referenceGradient.compute(r.getAs[Vector](1), r.getAs[Vector](0)(label), model))
      .treeReduce((x, y) => (Vectors.fromBreeze(x._1.toBreeze + y._1.toBreeze), x._2 + y._2))
  }

  "MatrixLogisticGradient " should " produce same losses for point" in  {
    checkLossForSinglePoint(Vectors.dense(0.3, 0.1), Vectors.dense(1.0, 1.0, 0.0))
  }

  "MatrixLogisticGradient " should " produce same losses for all points" in  {
    multiClassData.select("label", "features").collect().foreach(r =>
      checkLossForSinglePoint(r.getAs[Vector](1), r.getAs[Vector](0)))
  }

  def checkLossForSinglePoint(point: Vector, labels: Vector): Unit = {
    val accumulatedGradient = DenseMatrix.zeros(3, 2)
    val accumulatedLoss = Vectors.zeros(3).toDense

    MatrixLBFGS.computeGradient(point, labels, weights, accumulatedGradient, accumulatedLoss)

    val reference1 = referenceGradient.compute(point, labels(0), hiddenModel)
    val reference2 = referenceGradient.compute(point, labels(1), hiddenModel2)
    val reference3 = referenceGradient.compute(point, labels(2), hiddenModel3)

    accumulatedLoss(0) should be(reference1._2)
    accumulatedLoss(1) should be(reference2._2)
    accumulatedLoss(2) should be(reference3._2)
  }

  "MatrixLogisticGradient " should " produce same gradient for point" in  {
    checkGradientForSinglePoint(Vectors.dense(0.3, 0.1), Vectors.dense(1.0, 1.0, 0.0))
  }


  "MatrixLogisticGradient " should " produce same gradient for all points" in  {
    multiClassData.select("label", "features").collect().foreach(r =>
      checkGradientForSinglePoint(r.getAs[Vector](1), r.getAs[Vector](0)))
  }

  def checkGradientForSinglePoint(point: Vector, labels: Vector): Unit = {
    val accumulatedGradient = DenseMatrix.zeros(3, 2)
    val accumulatedLoss = Vectors.zeros(3).toDense

    MatrixLBFGS.computeGradient(point, labels, weights, accumulatedGradient, accumulatedLoss)

    val reference1 = referenceGradient.compute(point, labels(0), hiddenModel)
    val reference2 = referenceGradient.compute(point, labels(1), hiddenModel2)
    val reference3 = referenceGradient.compute(point, labels(2), hiddenModel3)


    accumulatedGradient(0, 0) should be(reference1._1(0))
    accumulatedGradient(0, 1) should be(reference1._1(1))

    accumulatedGradient(1, 0) should be(reference2._1(0))
    accumulatedGradient(1, 1) should be(reference2._1(1))

    accumulatedGradient(2, 0) should be(reference3._1(0))
    accumulatedGradient(2, 1) should be(reference3._1(1))
  }

  "MatrixLogisticGradient " should " produce same gradient for dataset" in  {

    val result = matrixResult

    val accumulatedGradient = result._1
    val accumulatedLoss = result._2


    accumulatedGradient(0, 0) should be(reference1._1(0) +- delta)
    accumulatedGradient(0, 1) should be(reference1._1(1) +- delta)

    accumulatedGradient(1, 0) should be(reference2._1(0) +- delta)
    accumulatedGradient(1, 1) should be(reference2._1(1) +- delta)

    accumulatedGradient(2, 0) should be(reference3._1(0) +- delta)
    accumulatedGradient(2, 1) should be(reference3._1(1) +- delta)

  }

  "MatrixLogisticGradient " should " produce same loss for dataset" in  {

    val result = matrixResult

    val accumulatedGradient = result._1
    val accumulatedLoss = result._2


    accumulatedLoss(0) should be(reference1._2 +- delta)
    accumulatedLoss(1) should be(reference2._2 +- delta)
    accumulatedLoss(2) should be(reference3._2 +- delta)
  }

  "MatrixLBFGS " should " produce same final weights" in  {
    val trainedForMatrix: Map[String, Vector] =
      MatrixLBFGS.multiClassLBFGS(multiClassData, "features", "label", 10, 1E-4, 100, 13)

    val optimizer: LBFGS = new LBFGS(new LogisticGradient(), new SquaredL2Updater())
    val trainedDirectly = Map(
      "0" -> optimizer.optimize(multiclassRdd.map(x => x._2(0) -> x._1), Vectors.zeros(2)),
      "1" -> optimizer.optimize(multiclassRdd.map(x => x._2(1) -> x._1), Vectors.zeros(2)),
      "2" -> optimizer.optimize(multiclassRdd.map(x => x._2(2) -> x._1), Vectors.zeros(2))
    )

    trainedForMatrix.foreach(result => {
      cosineDistance(result._2, trainedDirectly(result._1)) should be <= delta
    })
  }


  "Regularization " should " eliminate irrelevant" in {

    val addRandom = functions.udf[Vector,Vector](x => {
      Vectors.dense(2 * ThreadLocalRandom.current().nextDouble() - 1.0, x.toArray :_*)
    })

    val withIrrelevant = multiClassData.withColumn(
      "features",
      addRandom(multiClassData("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val trainedForMatrix: Map[String, Vector] =
      MatrixLBFGS.multiClassLBFGS(withIrrelevant, "features", "label", 10, 1E-5, 100, 13, 0.03)

    val hiddenModels = Map(
      "0" -> hiddenModel,
      "1" -> hiddenModel2,
      "2" -> hiddenModel3
    )

    trainedForMatrix.foreach(pair => {
      val v = pair._2
      Math.abs(v(0)) should be <= 0.0
      Math.abs(v(1)) should be > 0.0
      Math.abs(v(2)) should be > 0.0

      cosineDistance(Vectors.dense(v.toArray.drop(1)), hiddenModels(pair._1)) should be <= 0.001
    })
  }

  "Regularization " should " eliminate correlated" in {

    val noise = 0.05

    val addCorrelated = functions.udf[Vector,Vector](x => {
      Vectors.dense(x.toArray(0) / 1.5  +  2 * noise * ThreadLocalRandom.current().nextDouble() - noise, x.toArray :_*)
    })

    val withCorrelated = multiClassData.withColumn(
      "features",
      addCorrelated(multiClassData("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val trainedForMatrix: Map[String, Vector] =
      MatrixLBFGS.multiClassLBFGS(withCorrelated, "features", "label", 10, 1E-4, 100, 13, 0.03)

    val hiddenModels = Map(
      "0" -> hiddenModel,
      "1" -> hiddenModel2,
      "2" -> hiddenModel3
    )

    trainedForMatrix.foreach(pair => {
      val v = pair._2
      Math.abs(v(0)) should be <= 0.0
      Math.abs(v(1)) should be > 0.0
      Math.abs(v(2)) should be > 0.0

      cosineDistance(Vectors.dense(v.toArray.drop(1)), hiddenModels(pair._1)) should be <= 0.001
    })
  }

  "Regularization " should " remove intercept by default" in {

    val addIntercept = functions.udf[Vector,Vector](Interceptor.appendBias)

    val withIntercept = multiClassData.withColumn(
      "features",
      addIntercept(multiClassData("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val trainedForMatrix: Map[String, Vector] =
      MatrixLBFGS.multiClassLBFGS(withIntercept, "features", "label", 10, 1E-4, 100, 13, 0.03)

    val hiddenModels = Map(
      "0" -> hiddenModel,
      "1" -> hiddenModel2,
      "2" -> hiddenModel3
    )

    trainedForMatrix.foreach(pair => {
      val v = pair._2
      Math.abs(v(0)) should be > 0.0
      Math.abs(v(1)) should be > 0.0
      Math.abs(v(2)) should be <= 0.0

      cosineDistance(Vectors.dense(v.toArray.dropRight(1)), hiddenModels(pair._1)) should be <= 0.001
    })
  }

  "Regularization " should " keep intercept if asked" in {

    val addIntercept = functions.udf[Vector,Vector](Interceptor.appendBias)

    val withIntercept = multiClassData.withColumn(
      "features",
      addIntercept(multiClassData("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val trainedForMatrix: Map[String, Vector] =
      MatrixLBFGS.multiClassLBFGS(withIntercept, "features", "label", 10, 1E-4, 100, 13, 0.03, regulaizeLast = false)

    val hiddenModels = Map(
      "0" -> hiddenModel,
      "1" -> hiddenModel2,
      "2" -> hiddenModel3
    )

    trainedForMatrix.foreach(pair => {
      val v = pair._2
      Math.abs(v(0)) should be > 0.0
      Math.abs(v(1)) should be > 0.0
      Math.abs(v(2)) should be > 0.0

      cosineDistance(Vectors.dense(v.toArray.dropRight(1)), hiddenModels(pair._1)) should be <= 0.001
    })
  }

}
