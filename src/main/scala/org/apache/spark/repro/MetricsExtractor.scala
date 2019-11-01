package org.apache.spark.repro

import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.odkl.ModelWithSummary
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.DataFrame

trait MetricsExtractor extends Params {
  val extractExpression = new Param[String](this, "extractExpression",
    "Optional SQL expression for transforming metrics before uploading to repro context")

  def setExtractExpression(value: String) : this.type = set(extractExpression, value)

  final def extract(model: ModelWithSummary[_]): Option[DataFrame] = {
    extractImpl(model)
      .map(data => get(extractExpression)
        .map(expression => {
          new SQLTransformer().setStatement(expression).transform(data)
        })
        .getOrElse(data))
  }

  protected def extractImpl(model: ModelWithSummary[_]): Option[DataFrame]
}
