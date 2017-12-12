package org.apache.spark.ml.odkl.texts

import java.io.StringReader

import org.apache.lucene.analysis.standard.UAX29URLEmailTokenizer
import org.apache.lucene.analysis.tokenattributes.{CharTermAttribute, TypeAttribute}
import org.apache.lucene.util.AttributeFactory

/**
  * Created by eugeny.malyutin on 28.07.16.
  */
object URLElimminatorUtil {
  def geURLTokenizer() = new UAX29URLEmailTokenizer(AttributeFactory.DEFAULT_ATTRIBUTE_FACTORY)

  def filterText(text: String, tokenizer: UAX29URLEmailTokenizer) = {
    val stringBuilder = new StringBuilder

    val reader = new StringReader(text.toLowerCase)
    tokenizer.clearAttributes()
    tokenizer.setReader(reader)
    var charTermAttribute = tokenizer.addAttribute(classOf[CharTermAttribute]);
    var typeAttribute = tokenizer.addAttribute(classOf[TypeAttribute])
    tokenizer.reset()
    while (tokenizer.incrementToken()) {
      if (typeAttribute.`type`() != UAX29URLEmailTokenizer.TOKEN_TYPES(UAX29URLEmailTokenizer.URL)) {
        stringBuilder.++=(" " + charTermAttribute.toString)
      }
    }

    tokenizer.close()
    reader.close()
    stringBuilder.toString.trim
  }
}
