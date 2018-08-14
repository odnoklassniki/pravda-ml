package org.apache.spark.ml.odkl

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg
import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions
import org.scalatest.FlatSpec


/**
  * Created by vyacheslav.baranov on 02/12/15.
  */
class GridSearchSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with WithTestData with HasWeights with HasMetricsBlock {
  lazy val selectedModel = GridSearchSpec._selectedModel


  "GridSearch " should " train a model" in {

    val dev: linalg.Vector[Double] = Vectors.dense(hiddenModel.toArray ++ Array(0.0)).asBreeze - selectedModel.getCoefficients.asBreeze

    cosineDistance(Vectors.dense(Array(0.0) ++ hiddenModel.toArray), selectedModel.getCoefficients) should be <= delta
    selectedModel.getIntercept should be(0.0)
  }

  "Summary " should " add a model stat" in {
    val configurations = selectedModel.summary(Block("configurations"))
    configurations.count() should be(3 * 2)

    configurations.schema.size should be (5)

    configurations.schema(0).name should be("configurationIndex")
    configurations.schema(1).name should be("resultingMetric")
    configurations.schema(2).name should be("error")

    configurations.schema(3).name should endWith("elasticNetParam")
    configurations.schema(4).name should endWith("regParam")
  }

  "Summary " should " add configuration index to metrics" in {
    val data = selectedModel.summary(metrics)

    data.schema.fieldNames.toSet should contain("configurationIndex")
  }

  "Summary " should " add configuration index to weights" in {
    val data = selectedModel.summary(weights)

    data.schema.fieldNames.toSet should contain("configurationIndex")
  }

  "Best model " should " find that no regularization is better" in {
    val configurations = selectedModel.summary(Block("configurations"))

    configurations.first().getDouble(4) should be(0.0)
  }

  "Best model " should " have index 0" in {
    val configurations = selectedModel.summary(Block("configurations"))

    configurations.where("configurationIndex = 0").rdd.first().getDouble(1) should be(configurations.rdd.map(_.getDouble(1)).collect.max)
  }

  "Best model " should " have proper weights" in {
    val data = selectedModel.summary(weights)

    val bestWeights = data
      .where("configurationIndex = 0 AND foldNum = -1")
      .orderBy("name")
      .rdd
      .map(_.getAs[Double]("weight"))
      .collect()

    selectedModel.getCoefficients.toArray should be(bestWeights)
  }
}

object GridSearchSpec extends WithTestData {
  lazy val _selectedModel = {
    val addRandom = functions.udf[Vector, Vector](x => {
      Vectors.dense(
        2 * ThreadLocalRandom.current().nextDouble() - 1.0,
        x.toArray : _*)
    })

    val withIrrelevant = noInterceptDataLogistic.withColumn(
      "features",
      addRandom(noInterceptDataLogistic("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val nested = new LogisticRegressionLBFSG()


    val evaluated = Evaluator.crossValidate(
      nested,
      new TrainTestEvaluator(new BinaryClassificationEvaluator()),
      numFolds = 3,
      numThreads = 4)

    val paramGrid = new ParamGridBuilder()
      .addGrid(nested.regParam, Array(0.1, 0.01, 0.0))
      .addGrid(nested.elasticNetParam, Array(0.1, 0.0))

    val estimator = UnwrappedStage.cacheAndMaterialize(new GridSearch(evaluated)
      .setEstimatorParamMaps(paramGrid.build())
      .setMetricsExpression("SELECT AVG(value) FROM __THIS__ WHERE metric = 'auc' AND NOT isTest"))

    estimator.fit(withIrrelevant)
  }
}
