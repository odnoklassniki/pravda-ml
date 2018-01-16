package org.apache.spark.ml.odkl.texts

import java.io.StringReader

import org.apache.lucene.analysis.ar.ArabicAnalyzer
import org.apache.lucene.analysis.bg.BulgarianAnalyzer
import org.apache.lucene.analysis.br.BrazilianAnalyzer
import org.apache.lucene.analysis.ca.CatalanAnalyzer
import org.apache.lucene.analysis.cjk.CJKAnalyzer
import org.apache.lucene.analysis.cz.CzechAnalyzer
import org.apache.lucene.analysis.da.DanishAnalyzer
import org.apache.lucene.analysis.de.GermanAnalyzer
import org.apache.lucene.analysis.el.GreekAnalyzer
import org.apache.lucene.analysis.en.EnglishAnalyzer
import org.apache.lucene.analysis.es.SpanishAnalyzer
import org.apache.lucene.analysis.eu.BasqueAnalyzer
import org.apache.lucene.analysis.fa.PersianAnalyzer
import org.apache.lucene.analysis.fi.FinnishAnalyzer
import org.apache.lucene.analysis.fr.FrenchAnalyzer
import org.apache.lucene.analysis.ga.IrishAnalyzer
import org.apache.lucene.analysis.gl.GalicianAnalyzer
import org.apache.lucene.analysis.hi.HindiAnalyzer
import org.apache.lucene.analysis.hu.HungarianAnalyzer
import org.apache.lucene.analysis.hy.ArmenianAnalyzer
import org.apache.lucene.analysis.id.IndonesianAnalyzer
import org.apache.lucene.analysis.it.ItalianAnalyzer
import org.apache.lucene.analysis.lv.LatvianAnalyzer
import org.apache.lucene.analysis.no.NorwegianAnalyzer
import org.apache.lucene.analysis.ro.RomanianAnalyzer
import org.apache.lucene.analysis.ru.RussianAnalyzer
import org.apache.lucene.analysis.sv.SwedishAnalyzer
import org.apache.lucene.analysis.th.ThaiAnalyzer
import org.apache.lucene.analysis.tokenattributes.{CharTermAttribute, OffsetAttribute}
import org.apache.lucene.analysis.tr.TurkishAnalyzer
import org.apache.lucene.analysis.util.StopwordAnalyzerBase

/**
  * Created by eugeny.malyutin on 28.07.16.
  * This object is created to wrap language detector functionality for re-using without depending on spark.ml.* (Transformer and etc)
  */
object LanguageAwareStemmerUtil {

  val languageAnalyzersMap = {
    Map[String, () => StopwordAnalyzerBase](
      "ar" -> { () => new ArabicAnalyzer() },
      "bg" -> { () => new BulgarianAnalyzer() },
      "br" -> { () => new BrazilianAnalyzer() },
      "ca" -> { () => new CatalanAnalyzer() },
      "ch" -> { () => new CJKAnalyzer() }, //Chinise,Japanese,Korean
      "ja" -> { () => new CJKAnalyzer() }, //Chinise,Japanese,Korean
      "ko" -> { () => new CJKAnalyzer() },
      "cz" -> { () => new CzechAnalyzer() },
      "da" -> { () => new DanishAnalyzer() },
      "de" -> { () => new GermanAnalyzer() },
      "el" -> { () => new GreekAnalyzer() },
      "en" -> { () => new EnglishAnalyzer() },
      "es" -> { () => new SpanishAnalyzer() },
      "eu" -> { () => new BasqueAnalyzer() },
      "fa" -> { () => new PersianAnalyzer() },
      "fi" -> { () => new FinnishAnalyzer() },
      "fr" -> { () => new FrenchAnalyzer() },
      "ga" -> { () => new IrishAnalyzer() },
      "gl" -> { () => new GalicianAnalyzer() },
      "hi" -> { () => new HindiAnalyzer() },
      "hu" -> { () => new HungarianAnalyzer() },
      "hy" -> { () => new ArmenianAnalyzer() },
      "id" -> { () => new IndonesianAnalyzer() },
      "it" -> { () => new ItalianAnalyzer() },
      "lv" -> { () => new LatvianAnalyzer() },
      "no" -> { () => new NorwegianAnalyzer() },
      "ro" -> { () => new RomanianAnalyzer() },
      "ru" -> { () => new RussianAnalyzer() },
      "sv" -> { () => new SwedishAnalyzer() },
      "th" -> { () => new ThaiAnalyzer() },
      "tr" -> { () => new TurkishAnalyzer() }
    )
  }

  def stemmString(text: String, analyzer: StopwordAnalyzerBase): Array[String] = {
    val reader = new StringReader(text.toLowerCase)

    val tokens = analyzer.tokenStream("text", reader)
    val charTermAttribute = tokens.addAttribute(classOf[CharTermAttribute])
    tokens.reset()
    var ansList = scala.collection.mutable.ArrayBuffer.empty[String]
    while (tokens.incrementToken()) {
      ansList.append(charTermAttribute.toString)
    }
    tokens.close()
    ansList.toArray[String]
  }

  def instantiateMap = languageAnalyzersMap.mapValues(_.apply())

}
