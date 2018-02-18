package org.apache.spark.ml.odkl.texts

import org.apache.lucene.analysis.standard.UAX29URLEmailTokenizer
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.StructType

/**
  * Created by eugeny.malyutin on 05.05.16.
  *
  * Transformer to remove URL's from text based on lucene UAX29URLEmailTokenizer
  * With given column inputColumn of StringType returns outputColumn of StringType with text filtered non-url
  */
class URLElimminator(override val uid: String) extends Transformer with HasInputCol with HasOutputCol {
  @transient lazy val urlTokenizer = {
    new ThreadLocal[UAX29URLEmailTokenizer]() {
      override def initialValue(): UAX29URLEmailTokenizer = URLElimminatorUtil.geURLTokenizer()
    }
  }
  val filterTextUDF = udf((text: String) => {
    //numbers,urls
    if (text == null) {
      null
    } else {
      val localUrlTokenizer = urlTokenizer.get()
      URLElimminatorUtil.filterText(text, localUrlTokenizer)
    }
  })

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  def this() = this(Identifiable.randomUID("URLEliminator"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.withColumn($(outputCol), filterTextUDF(dataset.col($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    schema
  }
}


