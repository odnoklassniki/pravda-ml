package org.apache.spark.ml.odkl.hyperopt

import java.util.Random
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger}

import breeze.linalg.DenseVector
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.param.shared.{HasMaxIter, HasSeed, HasTol}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext}

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

  def toPairFromRow(row : Row, column: String) : ParamPair[T] = ParamPair(param, row.getAs[T](column))
}

case class ConfigNumber(number: Int, config: ParamMap) {
  override def toString: String = s"config_$number"
}

class StochasticHyperopt[ModelIn <: ModelWithSummary[ModelIn]]
(
  nested: SummarizableEstimator[ModelIn],
  override val uid: String) extends ForkedEstimator[ModelIn, ConfigNumber, ModelIn](nested, uid) with HyperoptParams
  with HasMaxIter with HasTol with HasSeed {

  def this(nested: SummarizableEstimator[ModelIn]) = this(nested, Identifiable.randomUID("stochasticHyperopt"))

  val paramDomains = new Param[Seq[ParamDomainPair[_]]](this, "paramDomains",
  "Domains of the parameters to optimize")

  val nanReplacement = new DoubleParam(this, "nanReplacement",
  "Value to use as evaluation result in case if model evaluation failed.")

  val searchMode = new Param[HyperParamSearcherFactory](this, "searchMode",
    "How to search for parameters. See HyperParamSearcher for supported modes. Default is random.")

  val maxNoImproveIters = new IntParam(this, "maxNoImproveIters",
    "How many iterations without improvement is allowed.")

  val topKForTolerance = new IntParam(this, "topKForTolerance",
    "How many top models to take when checking for convergence.")

  val epsilonGreedy = new DoubleParam(this, "epsilonGreedy",
    "Probability to sample uniform vector instead of guided search.")


  setDefault(
    searchMode -> HyperParamSearcher.RANDOM,
    seed -> System.nanoTime())

  def setParamDomains(pairs: ParamDomainPair[_]*) : this.type = set(paramDomains, pairs)

  def setMaxIter(iters: Int) : this.type = set(maxIter, iters)

  def setNanReplacement(value: Double) : this.type = set(nanReplacement, value)

  def setSearchMode(value: HyperParamSearcherFactory) : this.type = set(searchMode, value)

  def setTol(value: Double) : this.type = set(tol, value)
  def setMaxNoImproveIters(value: Int) : this.type = set(maxNoImproveIters, value)
  def setTopKForTolerance(value: Int) : this.type = set(topKForTolerance, value)

  def setEpsilonGreedy(value: Double) : this.type = set(epsilonGreedy, value)

  def setSeed(value: Long) : this.type = set(seed, value)

  override def fit(dataset: Dataset[_]): ModelIn = {
    if (isDefined(pathForTempModels) && !isSet(seed)) {
      logWarning(s"Persisting models without seed set could lead to inefficient behavior during restoration process at $uid")
    }
    try {
      super.fit(dataset)
    } catch {
      case NonFatal(e) => logError(s"Exception while fitting at $uid: ${e.toString}")
        throw e
    }
  }

  override protected def failFast(key: ConfigNumber, triedIn: Try[ModelIn]): Try[ModelIn] = {
    if (triedIn.isFailure) {
      logError(s"Fitting at $uid failed for $key due to ${triedIn.failed.get}")
    }
    // Grid search can tollerate some invalid configs
    triedIn
  }

  def createConfigDF(config: ParamMap): DataFrame = ???

  override def fitFork(estimator: SummarizableEstimator[ModelIn], wholeData: Dataset[_], partialData: (ConfigNumber, DataFrame)): (ConfigNumber, Try[ModelIn]) =
    try {
      // Copy the nested estimator
      logInfo(s"Trying configuration ${partialData._1.config}")
      val (config, model) = super.fitFork(estimator.copy(partialData._1.config), wholeData, partialData)

      if (isDefined(pathForTempModels)) {
        // In order to support restoration of the config tested we need to add configurations block to the model summary
        (config, model.map(x => {
          val rankedModels = Seq(extractParamsAndQuality(config.config, x))
          extractBestModel(wholeData.sqlContext, Seq(config.config -> model), rankedModels)
        }))
      } else {
        (config, model)
      }
    } catch {
      // Make sure errors in estimator copying are reported as model training failure.
      case NonFatal(e) => (partialData._1, failFast(partialData._1, Failure(e)))
    }

  override def copy(extra: ParamMap): SummarizableEstimator[ModelIn] = new StochasticHyperopt[ModelIn](nested.copy(extra))

  override protected def createForkSource(dataset: Dataset[_]): ForkSource[ModelIn, ConfigNumber, ModelIn]
  = new ForkSource[ModelIn, ConfigNumber, ModelIn] {

    val random = new Random($(seed))
    val guide : HyperParamSearcher = $(searchMode).create($(paramDomains).map(_.domain), $(seed))

    def vectorToParams(vector : DenseVector[Double]) : ParamMap = {
      ParamMap($(paramDomains).zipWithIndex.map(x => x._1.toParamPair(vector(x._2))) :_*)
    }

    def paramsToVector(params : ParamMap) : DenseVector[Double] ={
      DenseVector($(paramDomains).map(x => x.toDouble(params)).toArray)
    }

    val sequenceGenerator = new AtomicInteger()

    val convergenceFlag = new AtomicBoolean(false)

    override def nextFork(): Option[(ConfigNumber, DataFrame)] = {
      val index = sequenceGenerator.getAndIncrement()
      if (!isConverged(index))
      {
        guide.synchronized(
          Some(ConfigNumber(index, vectorToParams(guide.sampleInitialParams())) -> dataset.toDF))
      } else {
        None
      }
    }

    private val results = mutable.ArrayBuffer[(ConfigNumber,Try[ModelIn], Double)]()

    override def consumeFork(key: ConfigNumber, modelTry: Try[ModelIn]): Option[(ConfigNumber, DataFrame)] = {

      val tuple = modelTry.map(model => {
        if (isDefined(pathForTempModels)) {
          val row = model.summary(configurations).collect().head


          val evaluation = row.getAs[Number]($(resultingMetricColumn)).doubleValue()

          val restoredParams: Seq[ParamPair[_]] = $(paramDomains).map(x => {
            val columnName: String = get(paramNames).flatMap(_.get(x.param))
              .getOrElse({
                logWarning(s"Failed to find column name for param ${x.param}, restoration might not work properly")
                row.schema.fieldNames.find(_.endsWith(x.param.name)).get
              })

            x.toPairFromRow(row, columnName)
          })

          logInfo(s"At $uid got evaluation $evaluation for confg $params")
          (ConfigNumber(key.number, ParamMap(restoredParams :_*)), Try(model), evaluation)
        } else {
          val (params, extractedModel, evaluation) = extractParamsAndQuality(key.config, model)
          (ConfigNumber(key.number, params), Try(extractedModel), evaluation)
        }
      }).getOrElse((key, modelTry, Double.NaN))

      results.synchronized(results += tuple)

      val index = sequenceGenerator.getAndIncrement()
      if (!isConverged(index))
      {
        guide.synchronized(
          Some({
            val sample = guide.sampleNextParams(paramsToVector(tuple._1.config), if (tuple._3.isNaN) $(nanReplacement) else tuple._3)
            ConfigNumber(index, vectorToParams(
            if(!isDefined(epsilonGreedy) || random.nextDouble() > $(epsilonGreedy)) sample else guide.sampleInitialParams())) -> dataset.toDF
          }))
      } else {
        convergenceFlag.set(true)
        None
      }
    }

    override def createResult(): ModelIn = {
      val accumulated = results.result()

      extractBestModel(dataset.sqlContext,
        accumulated.filter(_._2.isFailure).map(x => x._1.config -> x._2),
        accumulated.filter(_._2.isSuccess).map(x => (x._1.config, x._2.get, x._3)).sortBy(-_._3))
    }

    private def isConverged(index: Int): Boolean = {
      if (convergenceFlag.get()) {
        logDebug("Convergence detected by another thread")
        return true
      }

      if(index > $(maxIter)) {
        logInfo(s"Search converged at $uid due to max iterations limit ${$(maxIter)}")
        return true
      }

      if (isDefined(maxNoImproveIters) || isDefined(topKForTolerance)) {
        val evaluations = results.synchronized(results.view.map(x => x._1.number -> x._3).toArray).sortBy(x => -x._2)
        if (evaluations.nonEmpty) {
          val bestConfig = evaluations.head._1

          if (isDefined(maxNoImproveIters) && evaluations.view.map(_._1).max > bestConfig + $(maxNoImproveIters)) {
            logInfo(s"Search converged at $uid due to max iterations without improvement limit ${$(maxNoImproveIters)}, best config found at index $bestConfig")
            return true
          }

          if (isDefined(topKForTolerance) && evaluations.size >= $(topKForTolerance)) {
            val bestModels = evaluations.view.map(_._2).take($(topKForTolerance))
            if (bestModels.head - bestModels.last < $(tol)) {
              logInfo(s"Search converged at $uid due to too small improvement among top ${$(topKForTolerance)} models ${bestModels.head - bestModels.last}")
              return true
            }
          }
        }
      }

      false
    }
  }

  /**
    * Not used due to custom forks source
    */
  override protected def createForks(dataset: Dataset[_]): Seq[(ConfigNumber, DataFrame)] = ???

  /**
    * Not used due to custom forks source
    */
  override protected def mergeModels(sqlContext: SQLContext, models: Seq[(ConfigNumber, Try[ModelIn])]): ModelIn = ???
}
