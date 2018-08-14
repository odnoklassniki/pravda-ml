package org.apache.spark.ml.odkl

import java.io.File

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions
import org.scalatest.FlatSpec

import scala.collection.immutable.Iterable

/**
  * Created by dmitriybugaichenko on 26.01.16.
  */
class PipeliningSpec extends FlatSpec with TestEnv with org.scalatest.Matchers
  with SQLOperations with WithTestData with HasWeights with HasMetricsBlock {

  val first = functions.udf[Double, Vector](x => x(0) * 10000 + 3000)
  val second = functions.udf[Double, Vector](x => x(1) * 10 - 5)

  lazy val rawDataForPipeline = withTypeAndClass
    .withColumn("firstFeature", first(withTypeAndClass("features")))
    .withColumn("secondFeature", second(withTypeAndClass("features")))
    .drop("features")

  lazy val rawDataWithLabel = typedWithLabels
    .withColumn("firstFeature", first(withTypeAndClass("features")))
    .withColumn("secondFeature", second(withTypeAndClass("features")))
    .drop("features")

  val negative: String = "Disliked"

  lazy val model = new Pipeline().setStages(Array(
    new VectorAssembler()
      .setInputCols(Array("firstFeature", "secondFeature"))
      .setOutputCol("features"),
    CombinedModel.perType(
      UnwrappedStage.repartition(
        numPartitions = 2,
        partitionBy = Seq("type"),
        sortBy = Seq("type", "classes"),
        estimator = UnwrappedStage.project(
          columns = Seq("classes", "features"),
          estimator = UnwrappedStage.cacheAndMaterialize(
            CombinedModel.linearCombination(
              Evaluator.crossValidate(
                Scaler.scale(Interceptor.intercept(new LogisticRegressionLBFSG())),
                new TrainTestEvaluator(new BinaryClassificationEvaluator()),
                numFolds = 2,
                numThreads = 4
              )
            ).setClassesMap("Negative" -> negative).setClassesWeights(negative -> -1.0))
        ))
    ).setNumThreads(4)
  )).fit(rawDataForPipeline)

  lazy val pair = {
    val directory = new File(FileUtils.getTempDirectory, model.uid)
    try {
      model.save(directory.getAbsolutePath)

      val result = PipelineModel.read.context(sqlc).load(directory.getAbsolutePath)

      val summary: ModelSummary = ModelWithSummary.extractSummary(result)

      // To make sure they are hear even after files are deleted
      summary.blocks.values.foreach(_.cache.count())

      (result, summary)
    } finally {
      FileUtils.deleteDirectory(directory)
    }
  }

  lazy val reReadModel = pair._1
  lazy val summary = pair._2

  "Pipeline " should " be trained with metrics" in {
    val summary = ModelWithSummary.extractSummary(model)

    val aucs = summary.$(metrics).filter("metric = 'auc'").select("value").rdd.map(_.getDouble(0)).collect()

    aucs.size should be(20)
    aucs.foreach(auc => auc should be >= 0.9)
  }

  "Pipeline " should " be trained with weights summary" in {
    val summary = ModelWithSummary.extractSummary(model)

    val names = summary.$(weights).select("name").distinct().rdd.map(_.getString(0)).collect().sorted

    names should be(Seq("#intercept", "firstFeature", "secondFeature"))
  }

  "Pipeline " should " be able to predict" in {

    val params = ParamMap(model.stages.collectFirst {
      case selector: CombinedModel[_, _] =>
        selector.nested.map {
          case (_, combination: LinearCombinationModel[_]) => combination.predictCombinations -> Array(Map("Positive" -> 1.0, negative -> -1.0))
        }
    }.get.toSeq: _*)

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(rawDataWithLabel, params))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }

  "Pipeline " should " be able to predict after re-read" in {
    val model = reReadModel

    val params = ParamMap(model.stages.collectFirst {
      case selector: CombinedModel[_, _] =>
        selector.nested.map {
          case (_, combination: LinearCombinationModel[_]) => combination.predictCombinations -> Array(Map("Positive" -> 1.0, negative -> -1.0))
        }
    }.get.toSeq: _*)

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(rawDataWithLabel, params))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }

  "Pipeline " should " preserve aggregated weights block after re-read" in {

    summary.blocks.size should be(2)

    val weights = summary.blocks(this.weights)

    val types = weights.select("type").distinct().rdd.map(_.getString(0)).collect().sorted
    types should be(Seq("Direct", "Inverse"))

    val classes = weights.select("classes").distinct().rdd.map(_.getString(0)).collect().sorted
    classes should be(Seq(negative, "Positive"))

    val combinations = weights.select("type", "classes").distinct().rdd.map(x => x.getString(0) -> x.getString(1)).collect().sorted
    combinations should be(Seq(
      "Direct" -> negative,
      "Direct" -> "Positive",
      "Inverse" -> negative,
      "Inverse" -> "Positive"
    ))

    val directPositiveWeights = weights
      .filter("type = 'Direct' AND classes = 'Positive' AND index >= 0")
      .select(index, weight).rdd
      .collect()
      .map(x => x.getInt(0) -> x.getDouble(1))
      .toMap

    directPositiveWeights.size should be(2)
  }

  "Pipeline " should " preserve aggregated metrics block after re-read" in {
    summary.blocks.size should be(2)

    val aucs = summary.$(metrics).filter("metric = 'auc'").select("value").rdd.map(_.getDouble(0)).collect()

    aucs.size should be(20)
    aucs.foreach(auc => auc should be >= 0.9)
  }

  "Pipeline " should " persist parameters after re-read" in {
    val model = reReadModel

    val linearCombination: LinearCombinationModel[_] = model.stages.collectFirst {
      case selector: CombinedModel[_, _] =>
        selector.nested.map {
          case (_, combination: LinearCombinationModel[_]) => combination
        }
    }.get.head

    val classesWeights: Map[String, Double] = linearCombination.get(linearCombination.predictCombinations).get(0)

    classesWeights.size should be(1.0)
    classesWeights(negative) should be(-1.0)
  }
}
