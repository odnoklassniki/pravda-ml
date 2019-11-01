package org.apache.spark.repro

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.odkl.CrossValidator.FoldsAssigner
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.odkl.hyperopt.GridSearch
import org.apache.spark.ml.param.ParamMap
import org.mlflow.api.proto.Service
import org.mlflow.tracking.ActiveRun
import org.mockito.{ArgumentMatcher, ArgumentMatchers, Mockito}
import org.scalatest.{FlatSpec, Matchers}

class MlFLowContextSpec extends FlatSpec with Matchers with WithTestData {

  lazy val rootMlFlow = MlFLowContextSpec._rootMlFlow
  lazy val modelWithMetrics = MlFLowContextSpec._modelWithMetrics
  lazy val diveMocks = MlFLowContextSpec._foldDive
  lazy val configDive = MlFLowContextSpec._configDive
  lazy val configDiveRun = MlFLowContextSpec._configDiveRun

  "CrossValidator" should " train a model" in {
    modelWithMetrics shouldNot be(null)
  }


  "CrossValidator" should " report metrics for folds" in {
    diveMocks.take(3).foreach(x =>
      Mockito.verify(x._2).logMetricsBatch(ArgumentMatchers.argThat(new ArgumentMatcher[Iterable[Service.Metric]]() {
        override def matches(argument: Iterable[Service.Metric]): Boolean = {
          val map = argument.map(x => x.getKey -> x).toMap

          map.size should be(4)
          map("auc on test").getValue should be(1.0 +- 0.0001)
          map("auc_pr on test").getValue should be(1.0 +- 0.0001)
          map("auc on train").getValue should be(1.0 +- 0.0001)
          map("auc_pr on train").getValue should be(1.0 +- 0.0001)
          true
        }
      })))
  }

  "CrossValidator" should " report metrics for global model" in {
    Mockito.verify(diveMocks.last._2).logMetricsBatch(ArgumentMatchers.argThat(new ArgumentMatcher[Iterable[Service.Metric]]() {
      override def matches(argument: Iterable[Service.Metric]): Boolean = {
        val map = argument.map(x => x.getKey -> x).toMap

        map.size should be(2)
        map("auc on train").getValue should be(1.0 +- 0.0001)
        map("auc_pr on train").getValue should be(1.0 +- 0.0001)
        true
      }
    }))
  }


  "GridSearch" should " report metrics for global model" in {
    Mockito.verify(rootMlFlow).logMetricsBatch(ArgumentMatchers.argThat(new ArgumentMatcher[Iterable[Service.Metric]]() {
      override def matches(argument: Iterable[Service.Metric]): Boolean = {
        val map = argument.map(x => x.getKey -> x).toMap

        map.size should be(1)
        map("target").getValue should be(1.0 +- 0.0001)
        true
      }
    }))
  }

  "GridSearch" should " report metrics for config" in {
    Mockito.verify(configDive).logMetricsBatch(ArgumentMatchers.argThat(new ArgumentMatcher[Iterable[Service.Metric]]() {
      override def matches(argument: Iterable[Service.Metric]): Boolean = {
        val map = argument.map(x => x.getKey -> x).toMap

        map.size should be(2)
        map("auc").getValue should be(1.0 +- 0.0001)
        map("auc_pr").getValue should be(1.0 +- 0.0001)
        true
      }
    }))
  }

  "GridSearch" should " report params for config" in {
    Mockito.verify(configDiveRun).logParams(ArgumentMatchers.argThat(new ArgumentMatcher[java.util.Map[String,String]]() {
      override def matches(argument: java.util.Map[String,String]): Boolean = {
         argument.size == 1 && argument.get("regParam") == "0.001"
      }
    }))
  }
}

object MlFLowContextSpec extends WithTestData {
  lazy val _rootMlFlow = Mockito.mock(classOf[MLFlowClient])

  lazy val (_configDive, _configDiveRun) = {
    val forkMock = Mockito.mock(classOf[MLFlowClient])
    val forkRun = Mockito.mock(classOf[ActiveRun])
    Mockito.when(forkMock.run).thenReturn(forkRun)
    Mockito.when(_rootMlFlow.dive(s"Fork for configuration=0")).thenReturn(forkMock)
    forkMock -> forkRun
  }

  lazy val _foldDive = Seq(0, 1, 2, -1).map(i => {
    val forkMock = Mockito.mock(classOf[MLFlowClient])
    val forkRun = Mockito.mock(classOf[ActiveRun])
    Mockito.when(forkMock.run).thenReturn(forkRun)
    Mockito.when(_configDive.dive(s"Fork for fold=$i")).thenReturn(forkMock)
    forkMock -> forkMock
  })

  lazy val _modelWithMetrics = {
    import org.apache.spark.repro.ReproContext._
    import sqlc.implicits._

    Mockito.when(_rootMlFlow.run).thenReturn(Mockito.mock(classOf[ActiveRun]))
    _foldDive.size


    implicit lazy val _reproContext: MlFlowReproContext = new MlFlowReproContext(spark, "", _rootMlFlow, Seq())

    val logReg = new LogisticRegressionLBFSG()

    new Pipeline().setStages(Array(
      new VectorAssembler()
        .setInputCols(Array("first", "second"))
        .setOutputCol("features"),
      UnwrappedStage.cacheAndMaterialize(
        new GridSearch(Evaluator.addFolds(
          estimator = Evaluator.validateInFolds(
            estimator = Scaler.scale(Interceptor.intercept(logReg)),
            evaluator = new TrainTestEvaluator(new BinaryClassificationEvaluator()),
            numFolds = 3)
            .setExtractExpression(
              "SELECT metric, AVG(value) AS value FROM __THIS__ WHERE `x-value` IS NULL AND isTest GROUP BY metric"),
          folder = new FoldsAssigner().setNumFolds(3)))
          .setMetricsExpression("SELECT AVG(value) FROM __THIS__ WHERE metric = 'auc' AND isTest")
          .setEstimatorParamMaps(Array(ParamMap(logReg.regParam -> 0.001)))
          .setParamNames(logReg.regParam -> "regParam"))))
    .tracedFit(rawData.withColumn("label", logistic('label)))
  }
}
