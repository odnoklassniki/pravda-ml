package org.apache.spark.ml.odkl.texts

import com.google.common.base.Optional
import com.optimaize.langdetect.LanguageDetector
import com.optimaize.langdetect.i18n.LdLocale
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{StringType, StructType}

import scala.collection.Map

/**
  * Created by eugeny.malyutin on 05.05.16.
  *
  * LanguageDetector transformer from String Column with Text creates String column with language-code or (Unknown)
  * Bulded around optimaize langdetect
  **/
class LanguageDetectorTransformer(override val uid: String) extends Transformer
  with HasInputCol with HasOutputCol {

  val minimalConfidence = new DoubleParam(this, "minimalConfidence", "langdetect parameter e.g. probability lower treshold")

  val languagePriors = new Param[Map[String, Double]](
    this, "languagePriors", "adjust probabilities for languages using our a-priory knowledge."
  )

  val priorAlpha = new Param[Double](
    this, "priorAlpha", "smoothing parameter used to allow languages we haven't seen")

  setDefault(
    inputCol -> "text",
    outputCol -> "lang",
    minimalConfidence -> 0.9D,
    languagePriors -> Map(),
    priorAlpha -> 0.01)


  val languageDetection = udf((text: String) => {
    val langOptional: Optional[LdLocale] = languageDetectorWrapped.languageDetector.detect(text)
    if (langOptional.isPresent) langOptional.get().getLanguage else "Unknown"
  })

  def setPriorAlpha(value: Double): this.type = set(priorAlpha, value)

  def setLanguagePriors(values: (String, Double)*): this.type = set(languagePriors, values.toMap)

  /** @group setParam */
  def setMinConfidence(value: Double): this.type = set(minimalConfidence, value)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  def this() = this(Identifiable.randomUID("languageDetector"))

  override def transform(dataset: DataFrame): DataFrame = {
    dataset.withColumn($(outputCol), languageDetection(dataset.col($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer = {
    defaultCopy(extra)
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.appendColumn(schema, $(outputCol), StringType)
  }

  @transient object languageDetectorWrapped extends Serializable {
    val languageDetector: LanguageDetector =
      LanguageDetectorUtils.buildLanguageDetector(
        LanguageDetectorUtils.readListLangsBuiltIn(),
        $(minimalConfidence),
        $(languagePriors).toMap)
  }

}
