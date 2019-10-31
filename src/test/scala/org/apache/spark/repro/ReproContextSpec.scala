package org.apache.spark.repro

import java.nio.file.Files

import breeze.linalg
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.regression.odkl.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.{FlatSpec, Matchers}

class ReproContextSpec extends FlatSpec with Matchers with WithTestData {
  lazy val tempPath = ReproContextSpec._tempPath
  lazy val tempPathWithMetrics = ReproContextSpec._tempPathForMetrics
  lazy val simpleModel = ReproContextSpec._simpleModel
  lazy val modelWithMetrics = ReproContextSpec._modelWithMetrics

  import spark.implicits._


  "ReproContext" should " train a simple model" in  {

    simpleModel shouldNot be(null)

    val estimator = new LinearRegressionSGD()

    val model = simpleModel.stages.last.asInstanceOf[LinearModel[_]]

    val dev: linalg.Vector[Double] = hiddenModel.asBreeze - model.getCoefficients.asBreeze

    val deviation: Double = dev dot dev

    deviation should be <= delta
    model.getIntercept should be(0.0 +- delta)
  }

  "ReproContext" should "persist estimator" in  {
    simpleModel shouldNot be(null)

    val loaded = Pipeline.load(tempPath + "/estimator")

    loaded.getStages.length should be(2)
  }

  "ReproContext" should "persist model" in  {
    simpleModel shouldNot be(null)

    val loaded = PipelineModel.load(tempPath + "/model")

    loaded.stages.length should be(2)
  }

  "ReproContext" should "persist params" in  {
    simpleModel shouldNot be(null)

    val loaded = sqlc.read.parquet(tempPath + "/params")

    loaded.schema.fieldNames should contain theSameElementsAs Seq("path", "param", "value")

    loaded.as[(String,String,String)].collect() should contain theSameElementsAs Seq(
      ("inputCols", """["first","second"]""", "stage0_VectorAssembler"),
      ("outputCol", "\"features\"", "stage0_VectorAssembler"),
      ("regParam", "0.001", "stage1_LinearRegression")
    )
  }

  "ReproContext" should "persist params for model with metrics" in {
    modelWithMetrics shouldNot be(null)

    val loaded = sqlc.read.parquet(tempPathWithMetrics + "/params")

    loaded.schema.fieldNames should contain theSameElementsAs Seq("path", "param", "value")

    val actual = loaded.as[(String, String, String)].collect()
    Seq(
      ("inputCols", """["first","second"]""", "stage0_VectorAssembler"),
      ("outputCol", "\"features\"", "stage0_VectorAssembler"),
      ("testMarker","\"isTest\"","stage1_CachingTransformer/FoldsAssigner/CrossValidator/EvaluatingTransformer/TrainOnlyFilter"),
      ("storageLevel","\"StorageLevel(memory, deserialized, 1 replicas)\"","stage1_CachingTransformer"),
      ("materializeCached","true","stage1_CachingTransformer"),
      ("regParam", "0.001", "stage1_CachingTransformer/FoldsAssigner/CrossValidator/EvaluatingTransformer/TrainOnlyFilter/LinearRegression")
    ).foreach(part => actual should contain(part))
  }

  "ReproContext" should "persist metrics" in {
    modelWithMetrics shouldNot be(null)

    val loaded = sqlc.read.parquet(tempPathWithMetrics + "/metrics")

    loaded.schema.fieldNames should contain theSameElementsAs Seq("foldNum", "isTest", "metric", "value")

    loaded.select('metric.as[String]).distinct().collect() should contain allElementsOf Seq(
      "r2", "rmse", "explainedVariance", "meanAbsoluteError", "meanSquaredError")

    loaded.select('foldNum.as[Int]).distinct.collect() should contain theSameElementsAs Seq(-1, 0, 1, 2)

    loaded.filter('metric === "r2" && 'isTest === true).count() should be(3)

    val r2s = loaded.where("metric = 'r2' and isTest").select('value.as[Double]).collect()
    r2s.sum / r2s.length should be(1.0 +- delta)
  }
}

object ReproContextSpec extends WithTestData {
  lazy val _tempPath = Files.createTempDirectory("reproContext").toString

  import org.apache.spark.repro.ReproContext._

  lazy val _simpleModel = {

    implicit lazy val _reproContext: ReproContext = new SimpleReproContext(_tempPath)(spark)

    new Pipeline().setStages(Array(
      new VectorAssembler()
        .setInputCols(Array("first", "second"))
        .setOutputCol("features"),
      new LinearRegression().setRegParam(0.001))
    ).reproducableFit(rawData)
  }

  lazy val _tempPathForMetrics = Files.createTempDirectory("tracedContext").toString


  lazy val _modelWithMetrics = {

    implicit lazy val _reproContext = new SimpleReproContext(_tempPathForMetrics)(spark)

    new Pipeline().setStages(Array(
      new VectorAssembler()
        .setInputCols(Array("first", "second"))
        .setOutputCol("features"),
      UnwrappedStage.cacheAndMaterialize(
        Evaluator.crossValidate(
          estimator = new LinearRegression().setRegParam(0.001),
          evaluator = new TrainTestEvaluator(new RegressionEvaluator()),
          numFolds = 3))
    )).tracedFit(rawData)
  }
}
