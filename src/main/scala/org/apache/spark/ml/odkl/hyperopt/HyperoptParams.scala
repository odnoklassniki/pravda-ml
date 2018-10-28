package org.apache.spark.ml.odkl.hyperopt

import org.apache.spark.ml.odkl.ModelWithSummary
import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.ml.param._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext, functions}

import scala.util.Try

trait HyperoptParams {
  this: Params =>

  val configurations: Block = Block("configurations")
  val paramNames: Param[Map[Param[_], String]] = new Param[Map[Param[_], String]](
    this, "paramsFriendlyNames", "Names of the parameters to use in column names to store configs"
  )
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

  def setParamNames(value: (Param[_], String)*): this.type = set(paramNames, value.toMap)

  def getMetricsBlock: String = $(metricsBlock)

  def setMetricsBlock(value: String): this.type = set(metricsBlock, value)

  def getMetricsExpression: String = $(metricsExpression)

  def setMetricsExpression(value: String): this.type = set(metricsExpression, value)

  def getConfigurationIndexColumn: String = $(configurationIndexColumn)

  def setConfigurationIndexColumn(value: String): this.type = set(configurationIndexColumn, value)

  def getResultingMetricColumn: String = $(resultingMetricColumn)

  def setResultingMetricColumn(value: String): this.type = set(resultingMetricColumn, value)

  protected def extractParamsAndQuality[ModelIn <: ModelWithSummary[ModelIn]](params: ParamMap, model: ModelIn): (ParamMap, ModelIn, Double) = {
    val metrics = model.summary.blocks(Block($(metricsBlock)))

    val tableName = model.uid + "_metrics"
    val query = $(metricsExpression).replaceAll("__THIS__", tableName)

    metrics.createOrReplaceTempView(tableName)

    val quality = metrics.sqlContext.sql(query).rdd.map(_.getAs[Number](0)).collect().map(_.doubleValue()).sum

    (params, model, quality)
  }

  protected def extractBestModel[ModelIn <: ModelWithSummary[ModelIn]](sqlContext: SQLContext, failedModels: Seq[(ParamMap, Try[ModelIn])], rankedModels: Seq[(ParamMap, ModelIn, Double)]): ModelIn = {
    // Extract parameters to build config for
    val keys: Seq[Param[_]] = rankedModels.head._1.toSeq.map(_.param.asInstanceOf[Param[Any]]).sortBy(_.name)

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

          StructField(get(paramNames).map(_.getOrElse(x, x.toString())).getOrElse(x.toString()), dataType, true)
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
    }) ++ failedModels.filter(_._2.isFailure).map(x => {
      val params = x._1
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
      ).reduce(_ union _)).toMap ++ Map(configurations -> configurationBlock)


    bestModel.copy(nestedBlocks)
  }
}
