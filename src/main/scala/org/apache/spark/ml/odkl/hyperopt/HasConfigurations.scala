package org.apache.spark.ml.odkl.hyperopt
import org.apache.spark.ml.odkl.ModelWithSummary
import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.repro.MetricsExtractor
import org.apache.spark.repro.ReproContext.logMetircs
import org.apache.spark.sql.{DataFrame, functions}

/**
  * Common summary block to store history of the hyperparameters search.
  */
trait HasConfigurations extends Params with MetricsExtractor {
  val configurations: Block = Block("configurations")

  val configurationIndexColumn = new Param[String](this, "configurationIndexColumn",
    "Name of the column to store id of config for further analysis.")
  val resultingMetricColumn = new Param[String](this, "resultingMetricColumn",
    "Name of the column to store resulting metrics for further analysis.")
  val errorColumn = new Param[String](this, "errorColumn",
    "Name of the column to store text of the error if occurs.")

  def getConfigurationIndexColumn: String = $(configurationIndexColumn)

  def setConfigurationIndexColumn(value: String): this.type = set(configurationIndexColumn, value)

  def getResultingMetricColumn: String = $(resultingMetricColumn)

  def setResultingMetricColumn(value: String): this.type = set(resultingMetricColumn, value)

  def getErrorColumn: String = $(errorColumn)

  def setErrorColumn(value: String): this.type = set(errorColumn, value)

  setDefault(
    configurationIndexColumn -> "configurationIndex",
    resultingMetricColumn -> "resultingMetric",
    errorColumn -> "error"
  )


  protected def extractImpl(model: ModelWithSummary[_]) : Option[DataFrame] = {
    // Report only resulting metrics to the context assuming that detailed metrics
    // where reported by forks.
    model.summary.blocks.get(configurations).map(data => data.select(
        data(getConfigurationIndexColumn).as("invertedStep"),
        data(getResultingMetricColumn).as("value"),
        functions.lit("target").as("metric")
      )
    )
  }
}
