package org.apache.spark.ml.odkl.hyperopt

import breeze.linalg.DenseVector
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.hyperparameter.search.{GaussianProcessSearch, RandomSearch}

/**
  * Used to perform hyperparameters search via sampling. Sampling is performed from the [0,1]^^N hypercube
  * and mapping from [0,1] to particular values are made by the parameter domains. Availible implementations
  * of Bayessian optimizers are largely based on the LinkedIn Photon-ML project.
  */
trait BayesianParamOptimizer {

  /**
    * @return Initial random sample of parameters to investigate.
    */
  def sampleInitialParams(): DenseVector[Double]

  /**
    * @return Having evaluation for the certain parameters sample next point to evaluate.
    */
  def sampleNextParams(observation: DenseVector[Double], value: Double): DenseVector[Double]
}

/**
  * Creates optimizer given the problem description.
  */
trait BayesianParamOptimizerFactory {
  /**
    * @param domains Description of the parameters to search for. Used to perform mapping from [0,1] to actual
    *                parameter values.
    * @param seed    Seed for the sampling
    * @param prirors Prior evaluations (typically created by the previous model evaluation, eg. yesterday model)
    * @return Optimizer to use for sampling.
    */
  def create(domains: Seq[ParamDomain[_]], seed: Long, prirors: Option[Seq[(Double, DenseVector[Double])]]) : BayesianParamOptimizer
}

object BayesianParamOptimizer {
  /**
    * Simple sampler from the Sobol points sequence
    */
  lazy val RANDOM = new BayesianParamOptimizerFactory {
    override def create(domains: Seq[ParamDomain[_]], seed: Long, priors : Option[Seq[(Double, DenseVector[Double])]]): BayesianParamOptimizer =
      new RandomOptimizer(domains, seed, buildDiscreteMap(domains))

    override def toString: String = "RANDOM"
  }

  /**
    * Advanced sampler modeling the efficiency function as a family of Gaussian processes (with integrated
    * kernel parameters) and sampling from it trying to maximize Expected Improvement.
    */
  lazy val GAUSSIAN_PROCESS = new BayesianParamOptimizerFactory {
    override def create(domains: Seq[ParamDomain[_]], seed: Long, priors : Option[Seq[(Double, DenseVector[Double])]]): BayesianParamOptimizer =
      {
        val result = new GaussianProcessOptimizer(domains, seed, buildDiscreteMap(domains), noisyTarget = false)

        priors.foreach(_.foreach(x => result.onPriorObservation(x._2, x._1)))

        result
      }

    override def toString: String = "GAUSSIAN_PROCESS"
  }

  /**
    * Variation of the GP optimizer which asumes there is a noise in the evaluations. The noise might be introduced
    * by instabilities of the learning process for complex non-convex models (eg. neural networks).
    */
  lazy val GAUSSIAN_PROCESS_WITH_NOISE = new BayesianParamOptimizerFactory {
    override def create(domains: Seq[ParamDomain[_]], seed: Long, priors : Option[Seq[(Double, DenseVector[Double])]]): BayesianParamOptimizer =
    {
      val result = new GaussianProcessOptimizer(domains, seed, buildDiscreteMap(domains), noisyTarget = true)

      priors.foreach(_.foreach(x => result.onPriorObservation(x._2, x._1)))

      result
    }

    override def toString: String = "GAUSSIAN_PROCESS_WITH_NOISE"
  }

  /**
    * Used to hack LinkedIn a bit since we are pushing evaluations from outside in order to support parallel search.
    */
  private[hyperopt] lazy val USELESS_EVAL = new EvaluationFunction[Double] {
    override def apply(hyperParameters: DenseVector[Double]): (Double, Double) = ???
    override def convertObservations(observations: Seq[Double]): Seq[(DenseVector[Double], Double)] = ???
    override def vectorizeParams(result: Double): DenseVector[Double] = ???
    override def getEvaluationValue(result: Double): Double = ???
  }

  private[hyperopt] def buildDiscreteMap(domains: Seq[ParamDomain[_]]) : Map[Int,Int] = {
    domains.zipWithIndex.filter(_._1.numDiscreteValues.isDefined).map(
      x => x._2 -> x._1.numDiscreteValues.get
    ).toMap
  }
}

/**
  * Simple sampler from the Sobol points sequence
  */
class RandomOptimizer(domains: Seq[ParamDomain[_]], seed: Long, discreteParams: Map[Int,Int]) extends RandomSearch[Double](
  domains.size, BayesianParamOptimizer.USELESS_EVAL,
  seed = seed,
  discreteParams = discreteParams
) with BayesianParamOptimizer {

  override def sampleInitialParams(): DenseVector[Double] = drawCandidates(1)(0, ::).t

  override def sampleNextParams(observation: DenseVector[Double], value: Double): DenseVector[Double] =
  // Note that Photon-ML tries to MINIMIZE the functions while we want to MAXIMIZE it
  // Thus negate the value
    super.next(observation, -value)
}

/**
  * Advanced sampler modeling the efficiency function as a family of Gaussian processes (with integrated
  * kernel parameters) and sampling from it trying to maximize Expected Improvement.
  */
class GaussianProcessOptimizer(domains: Seq[ParamDomain[_]], seed: Long, discreteParams: Map[Int,Int], noisyTarget: Boolean) extends GaussianProcessSearch[Double](
  domains.size, BayesianParamOptimizer.USELESS_EVAL,
  seed = seed,
  discreteParams = discreteParams,
  noisyTarget = noisyTarget
) with BayesianParamOptimizer {

  override def onPriorObservation(point: DenseVector[Double], eval: Double): Unit =
  // Note that Photon-ML tries to MINIMIZE the functions while we want to MAXIMIZE it
  // Thus negate the value
    super.onPriorObservation(point, -eval)

  override def sampleInitialParams(): DenseVector[Double] = drawCandidates(1)(0, ::).t

  override def sampleNextParams(observation: DenseVector[Double], value: Double): DenseVector[Double] =
  // Note that Photon-ML tries to MINIMIZE the functions while we want to MAXIMIZE it
  // Thus negate the value
    super.next(observation, -value)
}