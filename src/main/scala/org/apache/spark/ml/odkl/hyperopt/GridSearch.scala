package org.apache.spark.ml.odkl.hyperopt

import org.apache.hadoop.io.MD5Hash
import org.apache.spark.annotation.Since
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.repro.ReproContext
import org.apache.spark.sql._

import scala.collection.mutable
import scala.util.control.NonFatal
import scala.util.{Failure, Try}

/**
  * Provides ability to search through multiple configurations in parallel mode, collecting all the stat
  * and metrics.
  *
  * Supports persisting temporary models in order to restore after failures, but only when used with
  * StableOrderParamGridBuilder.
  */
class GridSearch[ModelIn <: ModelWithSummary[ModelIn]]
(
  nested: SummarizableEstimator[ModelIn],
  override val uid: String) extends ForkedEstimator[ModelIn, ConfigHolder, ModelIn](nested, uid) with HyperparametersOptimizer[ModelIn] {

  def this(nested: SummarizableEstimator[ModelIn]) = this(nested, Identifiable.randomUID("gridSearch"))

  val estimatorParamMaps: Param[Array[ParamMap]] = new Param[Array[ParamMap]](
    this, "estimatorParamMaps", "All the configurations to test in grid search.") {
    override def jsonEncode(value: Array[ParamMap]): String = ""

    override def jsonDecode(json: String): Array[ParamMap] = super.jsonDecode(json)
  }

  def getEstimatorParamMaps: Array[ParamMap] = $(estimatorParamMaps)

  def setEstimatorParamMaps(value: Array[ParamMap]): this.type = set(estimatorParamMaps, value)

  /**
    * Override this method and create forks to train from the data.
    */
  override protected def createForks(dataset: Dataset[_]): Seq[(ConfigHolder, DataFrame)] = {
    val result = $(estimatorParamMaps).zipWithIndex.map(x => ConfigHolder(x._2, x._1) -> dataset.toDF())
    logInfo(s"Got ${result.length} configurations to search through")
    result
  }


  override protected def failFast(key: ConfigHolder, triedIn: Try[ModelIn]): Try[ModelIn] = {
    if (triedIn.isFailure) {
      logError(s"Fitting at $uid failed for $key due to ${triedIn.failed.get}")
    }
    // Grid search can tollerate some invalid configs
    triedIn
  }

  /**
    * Given models trained for each fork create a combined model. This model is the
    * result of the estimator.
    */
  override protected def mergeModels(sqlContext: SQLContext, models: Seq[(ConfigHolder, Try[ModelIn])]): ModelIn = {

    val goodModels = models.filter(_._2.isSuccess)

    // Rank models by their metrics
    val rankedModels: Seq[(ParamMap, ModelIn, Double)] = goodModels.map(x => {
      extractParamsAndQuality(x._1.config, x._2.get)
    }).sortBy(x => -x._3)

    extractBestModel(sqlContext, models.filter(_._2.isFailure).map(x => x._1.config -> x._2), rankedModels).setParent(this)
  }

  override def copy(extra: ParamMap): GridSearch[ModelIn] = copyValues(new GridSearch[ModelIn](nested.copy(extra)), extra)

  override def fitFork(estimator: SummarizableEstimator[ModelIn], wholeData: Dataset[_], partialData: (ConfigHolder, DataFrame)): (ConfigHolder, Try[ModelIn]) =
    try {
      // Copy the nested estimator
      super.fitFork(estimator.copy(partialData._1.config), wholeData, partialData)
    } catch {
      // Make sure errors in estimator copying are reported as model training failure.
      case NonFatal(e) => (partialData._1, failFast(partialData._1, Failure(e)))
    }

  private def copyParamPair[T](original: ParamPair[T], value : Any) : ParamPair[T] = {
    ParamPair(original.param, value.asInstanceOf[T])
  }

  override protected def extractConfig(row: Row): (Double, ParamMap) = {
    val evaluation = row.getAs[Number]($(resultingMetricColumn)).doubleValue()

    val pairs = $(estimatorParamMaps).head.toSeq.map(x => {
      val columnName: String = get(paramNames).flatMap(_.get(x.param))
        .getOrElse({
          logWarning(s"Failed to find column name for param ${x.param}, restoration might not work properly")
          row.schema.fieldNames.find(_.endsWith(x.param.name)).get
        })

      copyParamPair(x, row.get(row.schema.fieldIndex(columnName)))
    })

    val restoredParams = ParamMap(pairs : _*)
    
    (evaluation, restoredParams)
  }

  override protected def getForkTags(partialData: (ConfigHolder, DataFrame)): Seq[(String, String)]
  = Seq("configuration" -> partialData._1.number.toString)

  override protected def diveToReproContext(partialData: (ConfigHolder, DataFrame), estimator: SummarizableEstimator[ModelIn]): Unit = {
    ReproContext.dive(getForkTags(partialData))
    ReproContext.logParamPairs(partialData._1.config.toSeq, Seq())
  }
}

/**
  * Builder for a param grid used in grid search-based model selection.
  * This builder provdes stable order and thus might be reliablly used with persist temp models
  * feature
  */
class StableOrderParamGridBuilder {

  private val paramGrid = mutable.ArrayBuffer[(Param[_], Iterable[_])]()
  private val addedParams = mutable.Set.empty[Param[_]]

  private val filters = mutable.ArrayBuffer[ParamMap => Boolean]()


  /**
    * Sets the given parameters in this grid to fixed values.
    */
  def baseOn(paramMap: ParamMap): this.type = {
    baseOn(paramMap.toSeq: _*)
    this
  }

  /**
    * Sets the given parameters in this grid to fixed values.
    */
  def baseOn(paramPairs: ParamPair[_]*): this.type = {
    paramPairs.foreach { p =>
      addGrid(p.param.asInstanceOf[Param[Any]], Seq(p.value))
    }
    this
  }

  /**
    * Adds a param with multiple values (overwrites if the input param exists).
    */
  def addGrid[T](param: Param[T], values: Iterable[T]): this.type = {

    if (!addedParams.add(param)) {
      throw new IllegalArgumentException("Duplicate param added to grid " + param)
    }

    paramGrid += (param -> values)
    this
  }

  // specialized versions of addGrid for Java.

  /**
    * Adds a double param with multiple values.
    */
  def addGrid(param: DoubleParam, values: Array[Double]): this.type = {
    addGrid[Double](param, values)
  }

  /**
    * Adds an int param with multiple values.
    */
  def addGrid(param: IntParam, values: Array[Int]): this.type = {
    addGrid[Int](param, values)
  }

  /**
    * Adds a float param with multiple values.
    */
  def addGrid(param: FloatParam, values: Array[Float]): this.type = {
    addGrid[Float](param, values)
  }

  /**
    * Adds a long param with multiple values.
    */
  def addGrid(param: LongParam, values: Array[Long]): this.type = {
    addGrid[Long](param, values)
  }

  /**
    * Adds a boolean param with true and false.
    */
  @Since("1.2.0")
  def addGrid(param: BooleanParam): this.type = {
    addGrid[Boolean](param, Array(true, false))
  }

  /**
    * Used to suppress certain tree branches from search. Configuration must satisfy all the filters in order
    * to be included.
    * @param filter  Functions taking the whole configuration and telling if it is valid (true) or not (false)
    */
  def addFilter(filter: ParamMap => Boolean) : this.type  = {
    filters += filter
    this
  }

  /**
    * Builds and returns all combinations of parameters specified by the param grid.
    */
  @Since("1.2.0")
  def build(): Array[ParamMap] = {
    var paramMaps = Array(new ParamMap)
    paramGrid.foreach { case (param, values) =>
      val newParamMaps = values.flatMap { v =>
        paramMaps.map(_.copy.put(param.asInstanceOf[Param[Any]], v))
      }
      paramMaps = newParamMaps.toArray
    }
    paramMaps.filter(x => filters.view.indexWhere(f => !f(x)) < 0)
  }
}

case class ConfigHolder(number: Int, config: ParamMap) {
  override def toString: String = {
    // This part is stable and does not depend on UIDs of parents, which can cause collisions.
    val encoded = config.toSeq.map(x => x.param.name + "=" + x.value.toString).sorted.toString()

    // Ordered number of config (when used with StableOrderParamGridBuilder) and hash of config values
    // should provide reliable and stable name.
    number.toString + "_" + MD5Hash.digest(encoded).toString
  }
}