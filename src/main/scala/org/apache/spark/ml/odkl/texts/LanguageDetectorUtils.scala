package org.apache.spark.ml.odkl.texts

import java.util

import com.optimaize.langdetect.LanguageDetector
import com.optimaize.langdetect.i18n.LdLocale
import com.optimaize.langdetect.ngram.NgramExtractors
import com.optimaize.langdetect.profiles.{LanguageProfile, LanguageProfileReader}

import scala.collection.JavaConverters
import scala.collection.JavaConverters._

/**
  * Created by eugeny.malyutin on 28.07.16.
  */
object LanguageDetectorUtils {
  val additionalLanguages: util.List[LdLocale] = Seq(
    LdLocale.fromString("az"),
    LdLocale.fromString("hy"),
    LdLocale.fromString("ka"),
    LdLocale.fromString("kk"),
    LdLocale.fromString("ky"),
    LdLocale.fromString("tg"),
    LdLocale.fromString("tk"),
    LdLocale.fromString("uz")
  ).asJava

  def readListLangsBuiltIn(): util.List[LanguageProfile] = {
    val reader =  new LanguageProfileReader()
    val builtIn = reader.readAllBuiltIn()
    builtIn.addAll(reader.readBuiltIn(additionalLanguages))
    builtIn
  }

  def buildLanguageDetector(listLangs: util.List[LanguageProfile], minimalConfidence: java.lang.Double, languagePriors: java.util.Map[String, java.lang.Double]): LanguageDetector = {
    buildLanguageDetector(listLangs,
      minimalConfidence.doubleValue(),
      languagePriors.asScala.mapValues(_.doubleValue()).toMap)
  }

  def buildLanguageDetector(listLangs: util.List[LanguageProfile], minimalConfidence: Double, languagePriors: Map[String, Double]): LanguageDetector = {
    val priorsMap: Map[LdLocale, Double] = JavaConverters.asScalaBufferConverter(listLangs).asScala
      .map(x => x.getLocale -> languagePriors.getOrElse(x.getLocale.getLanguage, 0.01))
      .toMap

    com.optimaize.langdetect.LanguageDetectorBuilder.create(NgramExtractors.standard())
      .withProfiles(listLangs)
      .languagePriorities(JavaConverters.mapAsJavaMapConverter(priorsMap.mapValues(_.asInstanceOf[java.lang.Double])).asJava)
      .minimalConfidence(minimalConfidence)
      .build()
  }
}
