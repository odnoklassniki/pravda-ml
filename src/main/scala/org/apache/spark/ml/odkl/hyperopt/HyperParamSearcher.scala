package org.apache.spark.ml.odkl.hyperopt

import breeze.linalg.DenseVector
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.hyperparameter.search.{GaussianProcessSearch, RandomSearch}
import org.apache.spark.ml.odkl._

trait HyperParamSearcher {

  def sampleInitialParams(): DenseVector[Double]

  def sampleNextParams(observation: DenseVector[Double], value: Double): DenseVector[Double]
}

trait HyperParamSearcherFactory {
  def create(domains: Seq[ParamDomain[_]], seed: Long) : HyperParamSearcher
}

object HyperParamSearcher {
  lazy val RANDOM = new HyperParamSearcherFactory {
    override def create(domains: Seq[ParamDomain[_]], seed: Long): HyperParamSearcher =
      new RandomSearcher(domains, seed)
  }

  lazy val GAUSSIAN_PROCESS = new HyperParamSearcherFactory {
    override def create(domains: Seq[ParamDomain[_]], seed: Long): HyperParamSearcher =
      new GaussianProcessSearcher(domains, seed)
  }
}

class RandomSearcher(domains: Seq[ParamDomain[_]], seed: Long) extends RandomSearch[Double](
  domains.size, new EvaluationFunction[Double] {
    override def apply(hyperParameters: DenseVector[Double]): (Double, Double) = ???
    override def convertObservations(observations: Seq[Double]): Seq[(DenseVector[Double], Double)] = ???
    override def vectorizeParams(result: Double): DenseVector[Double] = ???
    override def getEvaluationValue(result: Double): Double = ???
  },
  seed = seed
) with HyperParamSearcher {

  override def sampleInitialParams(): DenseVector[Double] = drawCandidates(1)(0, ::).t

  override def sampleNextParams(observation: DenseVector[Double], value: Double): DenseVector[Double] =
  // Note that Photon-ML tries to MINIMIZE the functions while we want to MAXIMIZE it
  // Thus negate the value
    super.next(observation, -value)
}

class GaussianProcessSearcher(domains: Seq[ParamDomain[_]], seed: Long) extends GaussianProcessSearch[Double](
  domains.size, new EvaluationFunction[Double] {
    override def apply(hyperParameters: DenseVector[Double]): (Double, Double) = ???
    override def convertObservations(observations: Seq[Double]): Seq[(DenseVector[Double], Double)] = ???
    override def vectorizeParams(result: Double): DenseVector[Double] = ???
    override def getEvaluationValue(result: Double): Double = ???
  },
  seed = seed
) with HyperParamSearcher {

  override def sampleInitialParams(): DenseVector[Double] = drawCandidates(1)(0, ::).t

  override def sampleNextParams(observation: DenseVector[Double], value: Double): DenseVector[Double] =
  // Note that Photon-ML tries to MINIMIZE the functions while we want to MAXIMIZE it
  // Thus negate the value
    super.next(observation, -value)
}