package org.apache.spark.ml.odkl

import org.apache.hadoop.io.MD5Hash
import org.apache.spark.annotation.Since
import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.types._

import scala.collection.mutable
import scala.util.control.NonFatal
import scala.util.{Failure, Try}

case class ConfigHolder(number: Int, config: ParamMap) {
  override def toString: String = {
    // This part is stable and does not depend on UIDs of parents, which can cause collisions.
    val encoded = config.toSeq.map(x => x.param.name + "=" + x.value.toString).sorted.toString()

    // Ordered number of config (when used with StableOrderParamGridBuilder) and hash of config values
    // should provide reliable and stable name.
    number.toString + "_" + MD5Hash.digest(encoded).toString
  }
}

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
  override val uid: String) extends ForkedEstimator[ModelIn, ConfigHolder, ModelIn](nested, uid) {

  def this(nested: SummarizableEstimator[ModelIn]) = this(nested, Identifiable.randomUID("gridSearch"))

  val configurations: Block = Block("configurations")

  val estimatorParamMaps: Param[Array[ParamMap]] =
    new Param(this, "estimatorParamMaps", "All the configurations to test in grid search.")

  val metricsBlock = new Param[String](this, "metricsBlock", "Name of the block with metrics to get results from.")

  val metricsExpression = new Param[String](this, "metricsExpression",
    "Expression used to extract single metric value from the metrics table. __THIS__ shoud be used as a table alias.")

  val configurationIndexColumn = new Param[String](this, "configurationIndexColumn",
    "Name of the column to store id of config for further analysis.")

  val resultingMetricColumn = new Param[String](this, "resultingMetricColumn",
    "Name of the column to store resulting metrics for further analysis.")

  val errorColumn = new Param[String](this, "errorColumn",
    "Name of the column to store text of the error if occurs.")

  setDefault(
    metricsBlock -> "metrics",
    configurationIndexColumn -> "configurationIndex",
    resultingMetricColumn -> "resultingMetric",
    errorColumn -> "error"
  )

  def getEstimatorParamMaps: Array[ParamMap] = $(estimatorParamMaps)

  def setEstimatorParamMaps(value: Array[ParamMap]): this.type = set(estimatorParamMaps, value)

  def getMetricsBlock: String = $(metricsBlock)

  def setMetricsBlock(value: String): this.type = set(metricsBlock, value)

  def getMetricsExpression: String = $(metricsExpression)

  def setMetricsExpression(value: String): this.type = set(metricsExpression, value)

  def getConfigurationIndexColumn: String = $(configurationIndexColumn)

  def setConfigurationIndexColumn(value: String): this.type = set(configurationIndexColumn, value)

  def getResultingMetricColumn: String = $(resultingMetricColumn)

  def setResultingMetricColumn(value: String): this.type = set(resultingMetricColumn, value)

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
      val params = x._1.config
      val model = x._2.get

      val metrics = model.summary.blocks(Block($(metricsBlock)))

      val tableName = model.uid + "_metrics"
      val query = $(metricsExpression).replaceAll("__THIS__", tableName)

      metrics.createOrReplaceTempView(tableName)

      val quality = metrics.sqlContext.sql(query).rdd.map(_.getAs[Number](0)).collect().map(_.doubleValue()).sum

      (params, model, quality)
    }).sortBy(x => -x._3)

    // Extract parameters to build config for
    val keys: Seq[Param[_]] = models.head._1.config.toSeq.map(_.param.asInstanceOf[Param[Any]]).sortBy(_.name)

    // Infer dataset schema
    val schema = StructType(
      Seq(
        StructField($(configurationIndexColumn), IntegerType),
        StructField($(resultingMetricColumn), DoubleType),
        StructField($(errorColumn), StringType)) ++
        keys.map(x => {
          val dataType = x match {
            case _: IntParam => IntegerType
            case _: DoubleParam => DoubleType
            case _: LongParam => LongType
            case _: BooleanParam => BooleanType
            case _: FloatParam => FloatType
            case _: StringArrayParam => ArrayType(StringType, true)
            case _: DoubleArrayParam => ArrayType(DoubleType, true)
            case _: IntArrayParam => ArrayType(IntegerType, true)
            case _ => StringType
          }

          StructField(x.toString(), dataType, true)
        }))

    def extractParams(params: ParamMap) = {
      keys.map(key => params.get(key).map(value => key match {
        case _: IntParam | _: DoubleParam | _: LongParam | _: BooleanParam | _: FloatParam => value
        case _: StringArrayParam | _: DoubleArrayParam | _: IntArrayParam => value
        case _ => key.asInstanceOf[Param[Any]].jsonEncode(value)
      }).get)
    }

    // Construct resulting block with variable part of configuration
    val rows = rankedModels.zipWithIndex.map(x => {
      val index: Int = x._2
      val params: ParamMap = x._1._1
      val metric: Double = x._1._3

      Row.fromSeq(Seq[Any](index, metric, "") ++ extractParams(params))
    }) ++ models.filter(_._2.isFailure).map(x => {
      val params = x._1.config
      val error = x._2.failed.get.toString

      Row.fromSeq(Seq[Any](Int.MaxValue, Double.NaN, error) ++ extractParams(params))
    })

    val configurationBlock = sqlContext.createDataFrame(
      sqlContext.sparkContext.parallelize(rows, 1),
      schema)


    // Now get the best model and enrich its summary
    val bestModel = rankedModels.head._2

    val nestedBlocks: Map[Block, DataFrame] = bestModel.summary.blocks.keys.map(
      block => block -> rankedModels.zipWithIndex.map(
        x => x._1._2.summary(block).withColumn($(configurationIndexColumn), functions.lit(x._2))
      ).reduce(_ unionAll _)).toMap ++ Map(configurations -> configurationBlock)


    bestModel.copy(nestedBlocks).setParent(this)
  }

  override def copy(extra: ParamMap): SummarizableEstimator[ModelIn] = new GridSearch[ModelIn](nested.copy(extra))

  override def fitFork(estimator: SummarizableEstimator[ModelIn], wholeData: Dataset[_], partialData: (ConfigHolder, DataFrame)): (ConfigHolder, Try[ModelIn]) =
    try {
      // Copy the nested estimator
      super.fitFork(estimator.copy(partialData._1.config), wholeData, partialData)
    } catch {
      // Make sure errors in estimator copying are reported as model training failure.
      case NonFatal(e) => (partialData._1, failFast(partialData._1, Failure(e)))
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
    paramMaps
  }
}