package org.apache.spark.ml.odkl.hyperopt

import java.util.concurrent.atomic.AtomicInteger

import breeze.linalg.DenseVector
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.param.shared.HasMaxIter
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamPair}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}

import scala.collection.mutable
import scala.util.control.NonFatal
import scala.util.{Failure, Try}

trait ParamDomain[T] {
  def toDouble(domain: T) : Double

  def fromDouble(double: Double) : T
}

class DoubleRangeDomain(lower: Double, upper: Double) extends ParamDomain[Double] {
  override def toDouble(domain: Double): Double = (domain - lower) / (upper - lower)

  override def fromDouble(double: Double): Double = double * (upper - lower) + lower
}

class ExponentialDoubleRangeDomain(lower: Double, upper: Double, base : Double = Math.E)
  extends DoubleRangeDomain(Math.pow(base, lower), Math.pow(base, upper)) {

  private val baseLog = Math.log(base)

  override def toDouble(domain: Double): Double = super.toDouble(Math.pow(base, domain))

  override def fromDouble(double: Double): Double = Math.log(super.fromDouble(double)) / baseLog
}

case class ParamDomainPair[T](param: Param[T], domain: ParamDomain[T]) {
  def toDouble(paramMap: ParamMap) : Double = domain.toDouble(paramMap.get(param).get)

  def toParamPair(double: Double) : ParamPair[T] = ParamPair(param, domain.fromDouble(double))
}

class StochasticHyperopt[ModelIn <: ModelWithSummary[ModelIn]]
(
  nested: SummarizableEstimator[ModelIn],
  override val uid: String) extends ForkedEstimator[ModelIn, ConfigHolder, ModelIn](nested, uid) with HyperoptParams
  with HasMaxIter {

  def this(nested: SummarizableEstimator[ModelIn]) = this(nested, Identifiable.randomUID("stochasticHyperopt"))

  val paramDomains = new Param[Seq[ParamDomainPair[_]]](this, "paramDomains",
  "Domains of the parameters to optimize")

  val nanReplacement = new DoubleParam(this, "nanReplacement",
  "Value to use as evaluation result in case if model evaluation failed.")

  val searchMode = new Param[HyperParamSearcherFactory](this, "searchMode",
    "How to search for parameters. See HyperParamSearcher for supported modes. Default is random.")

  setDefault(searchMode -> HyperParamSearcher.RANDOM)

  def setParamDomains(pairs: ParamDomainPair[_]*) : this.type = set(paramDomains, pairs)

  def setMaxIter(iters: Int) : this.type = set(maxIter, iters)

  def setNanReplacement(value: Double) : this.type = set(nanReplacement, value)

  def setSearchMode(value: HyperParamSearcherFactory) : this.type = set(searchMode, value)

  override protected def failFast(key: ConfigHolder, triedIn: Try[ModelIn]): Try[ModelIn] = {
    if (triedIn.isFailure) {
      logError(s"Fitting at $uid failed for $key due to ${triedIn.failed.get}")
    }
    // Grid search can tollerate some invalid configs
    triedIn
  }

  override def fitFork(estimator: SummarizableEstimator[ModelIn], wholeData: Dataset[_], partialData: (ConfigHolder, DataFrame)): (ConfigHolder, Try[ModelIn]) =
    try {
      // Copy the nested estimator
      super.fitFork(estimator.copy(partialData._1.config), wholeData, partialData)
    } catch {
      // Make sure errors in estimator copying are reported as model training failure.
      case NonFatal(e) => (partialData._1, failFast(partialData._1, Failure(e)))
    }

  override def copy(extra: ParamMap): SummarizableEstimator[ModelIn] = new StochasticHyperopt[ModelIn](nested.copy(extra))

  override protected def createForkSource(dataset: Dataset[_]): ForkSource[ModelIn, ConfigHolder, ModelIn]
  = new ForkSource[ModelIn, ConfigHolder, ModelIn] {

    val random : HyperParamSearcher = $(searchMode).create($(paramDomains).map(_.domain))

    def vectorToParams(vector : DenseVector[Double]) : ParamMap = {
      ParamMap($(paramDomains).zipWithIndex.map(x => x._1.toParamPair(vector(x._2))) :_*)
    }

    def paramsToVector(params : ParamMap) : DenseVector[Double] ={
      DenseVector($(paramDomains).map(x => x.toDouble(params)).toArray)
    }

    val sequenceGenerator = new AtomicInteger()

    override def nextFork(): Option[(ConfigHolder, DataFrame)] = {
      val index = sequenceGenerator.getAndIncrement()
      if (index < $(maxIter))
      {
        random.synchronized(
          Some(ConfigHolder(index, vectorToParams(random.sampleInitialParams())) -> dataset.toDF))
      } else {
        None
      }
    }

    private val results = mutable.ArrayBuilder.make[(ParamMap,Try[ModelIn], Double)]()

    override def consumeFork(key: ConfigHolder, modelTry: Try[ModelIn]): Option[(ConfigHolder, DataFrame)] = {

      val tuple = modelTry.map(model => {
        val (params, extractedModel, evaluation) = extractParamsAndQuality(key.config, model)
        (params, Try(extractedModel), evaluation)
      }).getOrElse((key.config, modelTry, Double.NaN))

      results.synchronized(results += tuple)

      val index = sequenceGenerator.getAndIncrement()
      if (index < $(maxIter))
      {
        random.synchronized(
          Some(ConfigHolder(index, vectorToParams(
            random.sampleNextParams(paramsToVector(tuple._1), if(tuple._3.isNaN) $(nanReplacement) else tuple._3))) -> dataset.toDF))
      } else {
        None
      }
    }

    override def createResult(): ModelIn = {
      val accumulated = results.result()

      extractBestModel(dataset.sqlContext,
        accumulated.filter(_._2.isFailure).map(x => x._1 -> x._2),
        accumulated.filter(_._2.isSuccess).map(x => (x._1, x._2.get, x._3)).sortBy(-_._3))
    }
  }

  /**
    * Not used due to custom forks source
    */
  override protected def createForks(dataset: Dataset[_]): Seq[(ConfigHolder, DataFrame)] = ???

  /**
    * Not used due to custom forks source
    */
  override protected def mergeModels(sqlContext: SQLContext, models: Seq[(ConfigHolder, Try[ModelIn])]): ModelIn = ???
}
