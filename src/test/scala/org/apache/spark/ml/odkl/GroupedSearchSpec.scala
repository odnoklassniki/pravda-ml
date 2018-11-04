package org.apache.spark.ml.odkl

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.ml.odkl.hyperopt._
import org.apache.spark.ml.param.ParamMap
import org.scalatest.FlatSpec


/**
  * Created by vyacheslav.baranov on 02/12/15.
  */
class GroupedSearchSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with WithTestData with HasWeights with HasMetricsBlock {
  private lazy val selectedModel = GroupedSearchSpec._selectedModel

  "Summary " should " add a model stat" in  {
    val configurations = selectedModel.summary(Block("configurations"))
    configurations.count() should be > 6L
    configurations.count() should be <= 20L

    configurations.schema.size should be(7)

    configurations.schema(0).name should be("configurationIndex")
    configurations.schema(1).name should be("resultingMetric")
    configurations.schema(2).name should be("error")

    configurations.schema(3).name should be("RegParam")

    configurations.schema(4).name should be("stageName")
    configurations.schema(5).name should be("stageReversedIndex")

    configurations.schema(6).name should be("ElasticNet")

    configurations.show(20, false)
  }

  "Summary " should " add configuration index to metrics" in  {
    val data = selectedModel.summary(metrics)

    data.schema.fieldNames.toSet should contain("configurationIndex")
  }

  "Summary " should " add configuration index to weights" in  {
    val data = selectedModel.summary(weights)

    data.schema.fieldNames.toSet should contain("configurationIndex")
  }

  "Best model " should " should have sound performance " in  {
    val configurations = selectedModel.summary(Block("configurations"))

    configurations.select("resultingMetric").rdd.map(_.getDouble(0)).collect().head should be >= 0.99
  }

  "Best model " should " have index 0" in  {
    val configurations = selectedModel.summary(Block("configurations"))

    configurations.where("configurationIndex = 0 AND stageReversedIndex == 0")
      .rdd.first().getDouble(1) should be(configurations.rdd.map(_.getDouble(1)).collect.max)
  }

  "Best model " should " have proper weights" in  {
    val data = selectedModel.summary(weights)

    val bestWeights = data
      .where("configurationIndex = 0 AND stageReversedIndex == 0 AND foldNum = -1")
      .orderBy("name")
      .rdd
      .map(_.getAs[Double]("weight"))
      .collect()

    selectedModel.getCoefficients.toArray should be(bestWeights)
  }
}

object GroupedSearchSpec extends WithTestData {
  lazy val _selectedModel  = {
    val nested = new LogisticRegressionLBFSG()

    val evaluated = Evaluator.crossValidate(
      nested,
      new TrainTestEvaluator(new BinaryClassificationEvaluator()),
      numFolds = 3,
      numThreads = 4)

    val optimizer = new StochasticHyperopt(evaluated)
      .setMaxIter(20)
      .setNanReplacement(0)
      .setSearchMode(BayesianParamOptimizer.GAUSSIAN_PROCESS)
      .setMaxNoImproveIters(6)
      .setTol(0.0005)
      .setTopKForTolerance(4)
      .setEpsilonGreedy(0.1)
      .setMetricsExpression("SELECT AVG(value) FROM __THIS__ WHERE metric = 'auc' AND isTest")
      .setParamNames(
        nested.regParam -> "RegParam"
      )

    val gridSearch = new GridSearch(evaluated)
      .setMetricsExpression("SELECT AVG(value) FROM __THIS__ WHERE metric = 'auc' AND isTest")
      .setParamNames(
        nested.elasticNetParam -> "ElasticNet"
      )

    val estimator = new GroupedSearch(Seq(
      "regParam" -> optimizer.copy(ParamMap(optimizer.paramDomains -> Seq(
        ParamDomainPair(nested.regParam, new DoubleRangeDomain(0, 2))))),
      "elasticNet" -> gridSearch.copy(ParamMap(gridSearch.estimatorParamMaps -> new StableOrderParamGridBuilder()
        .addGrid(nested.elasticNetParam, Array(0.0,0.25,0.5))
        .build()))
    ))

    UnwrappedStage.cacheAndMaterialize(estimator).fit(FeaturesSelectionSpec._withBoth)
  }
}


