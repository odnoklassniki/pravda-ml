package odkl.analysis.spark.texts

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.odkl.texts.LanguageDetectorTransformer
import org.scalatest.FlatSpec

class LanguageDetectorTransformerSpec extends FlatSpec with TestEnv with org.scalatest.Matchers {
  "LanguageDetector" should "detect language" in {
    import sqlc.implicits._
    val data = sc.parallelize(Seq(
      "Мальчик, водочки нам принеси! Мы домой летим",
      "Luke, I am Your Father",
      "սիրելի, ուրախ է տեսնել ձեզ"
    ))
      .toDF("text")

    val transformer = new LanguageDetectorTransformer()
        .setInputCol("text")
      .setOutputCol("lang")

    assertResult(Seq("ru", "en", "hy"))(transformer.transform(data)
        .collect()
      .map(_.getString(1)))
  }

  "Language Detector" should "work with language priors" in {
    import sqlc.implicits._
    val data = sc.parallelize(Seq("Прапор тобі в руки, барабан на шию, сокиру в спину і електричку назустріч\n"))
      .toDF("text")

    val transformer_1 = new LanguageDetectorTransformer()
      .setInputCol("text")
      .setOutputCol("lang")

    val transformer_2 = new LanguageDetectorTransformer()
      .setInputCol("text")
      .setOutputCol("lang")
        .setLanguagePriors( ("uk", 0.0))

    assertResult("uk")(transformer_1.transform(data).first().getString(1))
    assertResult("Unknown")(transformer_2.transform(data).first().getString(1))

  }
}
