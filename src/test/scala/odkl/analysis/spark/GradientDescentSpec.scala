package odkl.analysis.spark

import breeze.linalg.{DenseMatrix, DenseVector, Matrix, Vector, norm, sum}
import breeze.storage.Zero
import org.scalatest.{FlatSpec, Matchers}

import scala.annotation.tailrec
import scala.reflect.ClassTag

class GradientDescentSpec extends FlatSpec with Matchers{

  def gradientDescent(data: DenseMatrix[Double], target: Vector[Double]) : Vector[Double] = {
    var weights = DenseVector.zeros[Double](data.cols)
    var error: Vector[Double] = DenseVector.zeros[Double](data.rows)

    var i = 0

    do {
      error = target -:- (data * weights)
      val gradient = (1.0 / data.rows) * data.t * error

      weights = weights + gradient

      i += 1
    } while (i < 10000)

    weights
  }

  case class DescentParams(maxIters: Int = 10000, learningRate: Double = 1.0, tolerance: Double = 0.0001)

  @tailrec
  private final def step(data: DenseMatrix[Double], target: Vector[Double], weights: Vector[Double], num: Int, params: DescentParams) : Vector[Double] = {
    if(num > params.maxIters) {
      weights
    } else {
      val error = target -:- (data * weights)
      val gradient = (1.0 / data.rows) * data.t * error
      val newWeights = weights + params.learningRate * gradient

      if (norm(newWeights -:- weights) < params.tolerance) {
        newWeights
      } else {
        step(data, target, newWeights, num + 1, params)
      }                                                                        }
  }

  def recursiveDescent(data: DenseMatrix[Double], target: Vector[Double], params: DescentParams = DescentParams()) : Vector[Double] = {
     step(data, target, DenseVector.zeros[Double](data.cols), 1, params)
  }


  "Gradient descent" should "find model" in {
    val data = DenseMatrix.rand[Double](10000, 3)
    val model = DenseVector(0.1, -0.3, 0.5)

    val target = data * model

    val solution = gradientDescent(data, target)

    val value: DenseVector[Double] = model -:- solution

    norm(value) should be <= 0.01
  }

  "Gradient descent" should "find model recursively" in {
    val data = DenseMatrix.rand[Double](10000, 3)
    val model = DenseVector(0.1, -0.3, 0.5)

    val target = data * model

    val solution = recursiveDescent(data, target)

    val value: DenseVector[Double] = model -:- solution

    norm(value) should be <= 0.01
  }
}
