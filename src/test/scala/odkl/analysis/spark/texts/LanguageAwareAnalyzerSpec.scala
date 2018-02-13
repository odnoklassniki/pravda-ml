package odkl.analysis.spark.texts

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.odkl.texts.LanguageAwareAnalyzer
import org.scalatest.FlatSpec

class LanguageAwareAnalyzerSpec extends FlatSpec with TestEnv with org.scalatest.Matchers  {

  "LanguageAwareAnalyzer" should "stemm text by language" in {

    import sqlc.implicits._

    val dataFrame = sc.parallelize(Seq(
      ("ru", "Я помню чудное мгновенье"),
      ("en", "imagine there is no heaven It's easy if you try")
    ))
        .toDF("language", "text")

    val transformer = {
      new LanguageAwareAnalyzer()
        .setInputColLang("language")
        .setInputColText("text")
        .setOutputCol("stemmed")
    }
    val transformed = transformer.transform(dataFrame)
        .collect()

    val answers = Seq(Seq("помн", "чудн", "мгновен"),
      Seq("imagin", "heaven", "easi", "you", "try"))

    val stemmed =transformed.map(_.getSeq[String](2))
    assertResult(answers)(stemmed)

  }
  "LanguageAwareAnalyzer" should "stemm text without language by default" in {
    import sqlc.implicits._

    val dataFrame = sc.parallelize(Seq(
      ("unknown", "imagine there is no heaven It's easy if you try")
    ))
      .toDF("language", "text")

    val transformer = new LanguageAwareAnalyzer()
      .setInputColLang("language")
      .setInputColText("text")
      .setDefaultLanguage("en")
      .setOutputCol("stemmed")

    assertResult( Seq("imagin", "heaven", "easi", "you", "try"))(
      transformer.transform(dataFrame)
        .first()
        .getSeq[String](2)
    )
  }

}
