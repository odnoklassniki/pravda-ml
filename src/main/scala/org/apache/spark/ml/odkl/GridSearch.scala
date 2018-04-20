package org.apache.spark.ml.odkl

import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._
import org.apache.spark.sql._

class GridSearch[ModelIn <: ModelWithSummary[ModelIn]]
(
  nested: SummarizableEstimator[ModelIn],
  override val uid: String) extends ForkedEstimator[ModelIn, ParamMap, ModelIn](nested, uid) {

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

  setDefault(
    metricsBlock -> "metrics",
    configurationIndexColumn -> "configurationIndex",
    resultingMetricColumn -> "resultingMetric"
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
  override protected def createForks(dataset: Dataset[_]): Seq[(ParamMap, DataFrame)] = {
    $(estimatorParamMaps).map(x => (x, dataset.toDF()))
  }

  /**
    * Given models trained for each fork create a combined model. This model is the
    * result of the estimator.
    */
  override protected def mergeModels(sqlContext: SQLContext, models: Seq[(ParamMap, ModelIn)]): ModelIn = {

    // Rank models by their metrics
    val rankedModels = models.map(x => {
      val params = x._1
      val model = x._2

      val metrics = model.summary.blocks(Block($(metricsBlock)))

      val tableName = model.uid + "_metrics"
      val query = $(metricsExpression).replaceAll("__THIS__", tableName)

      metrics.createOrReplaceTempView(tableName)

      val quality = metrics.sqlContext.sql(query).rdd.map(_.getAs[Number](0)).collect().map(_.doubleValue()).sum

      (params, model, quality)
    }).sortBy(x => -x._3)

    // Extract parameters to build config for
    val keys: Seq[Param[_]] = rankedModels.head._1.toSeq.map(_.param.asInstanceOf[Param[Any]]).sortBy(_.name)

    // Infer dataset schema
    val schema = StructType(
      Seq(StructField($(configurationIndexColumn), IntegerType), StructField($(resultingMetricColumn), DoubleType)) ++
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

    // Construct resulting block with variable part of configuration
    val rows = rankedModels.zipWithIndex.map(x => {
      val index: Int = x._2
      val params = x._1._1
      val metric: Double = x._1._3

      Row.fromSeq(Seq[Any](index, metric) ++ keys.map(key => params.get(key).map(value => key match {
        case _: IntParam | _: DoubleParam | _: LongParam | _: BooleanParam | _: FloatParam => value
        case _: StringArrayParam | _: DoubleArrayParam | _: IntArrayParam => value
        case _ => key.asInstanceOf[Param[Any]].jsonEncode(value)
      }).get))
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

  override def fitFork(estimator: SummarizableEstimator[ModelIn], wholeData: Dataset[_], partialData: (ParamMap, DataFrame)): (ParamMap, ModelIn) =
    super.fitFork(estimator.copy(partialData._1), wholeData, partialData)
}
