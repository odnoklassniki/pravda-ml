package org.apache.spark.repro

import java.nio.file.Files
import java.{lang, util}

import breeze.linalg
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.regression.odkl.LinearRegression
import org.apache.spark.repro.ReproContextSpec.{rawData, spark}
import org.mlflow.tracking.ActiveRun
import org.mockito.{ArgumentMatcher, ArgumentMatchers, Mockito}
import org.scalatest.{FlatSpec, Matchers}

class MlFLowContextSpec extends FlatSpec with Matchers with WithTestData {

  lazy val modelWithMetrics = MlFLowContextSpec._modelWithMetrics
  lazy val diveMocks = MlFLowContextSpec._divedMocks

  "MlFLowContext" should " train a model" in  {
    modelWithMetrics shouldNot be(null)
  }


  "MlFLowContext" should " report metrics for folds" in  {
    diveMocks.take(3).foreach(x =>
    Mockito.verify(x._2).logMetrics(ArgumentMatchers.argThat(new ArgumentMatcher[java.util.Map[String, java.lang.Double]]() {
      override def matches(argument: util.Map[String, lang.Double]): Boolean = {
        argument.size() should be(4)
        argument.get("auc on test").toDouble should be(1.0 +- 0.0001)
        argument.get("auc_pr on test").toDouble should be(1.0 +- 0.0001)
        argument.get("auc on train").toDouble should be(1.0 +- 0.0001)
        argument.get("auc_pr on train").toDouble should be(1.0 +- 0.0001)
        true
      }
    })))
  }

  "MlFLowContext" should " report metrics for global model" in  {
      Mockito.verify(diveMocks.last._2).logMetrics(ArgumentMatchers.argThat(new ArgumentMatcher[java.util.Map[String, java.lang.Double]]() {
        override def matches(argument: util.Map[String, lang.Double]): Boolean = {
          argument.size() should be(2)
          argument.get("auc on train").toDouble should be(1.0 +- 0.0001)
          argument.get("auc_pr on train").toDouble should be(1.0 +- 0.0001)
          true
        }
      }))
  }
}

object MlFLowContextSpec extends WithTestData {
  lazy val _mlFlowMock = Mockito.mock(classOf[MLFlowClient])

  lazy val _divedMocks = Seq(0,1,2,-1).map(i => {
    val forkMock = Mockito.mock(classOf[MLFlowClient])
    val forkRun = Mockito.mock(classOf[ActiveRun])
    Mockito.when(forkMock.run).thenReturn(forkRun)
    Mockito.when(_mlFlowMock.dive(s"Fork for fork=$i")).thenReturn(forkMock)
    forkMock -> forkRun
  })

  lazy val _modelWithMetrics = {
    import org.apache.spark.repro.ReproContext._
    import sqlc.implicits._

    Mockito.when(_mlFlowMock.run).thenReturn(Mockito.mock(classOf[ActiveRun]))
    _divedMocks.size


    implicit lazy val _reproContext: MlFlowReproContext = new MlFlowReproContext(spark, "", _mlFlowMock, Seq())

    new Pipeline().setStages(Array(
      new VectorAssembler()
        .setInputCols(Array("first", "second"))
        .setOutputCol("features"),
      UnwrappedStage.cacheAndMaterialize(
        Evaluator.crossValidate(
          estimator = new LogisticRegressionLBFSG().setRegParam(0.001),
          evaluator = new TrainTestEvaluator(new BinaryClassificationEvaluator()),
          numFolds = 3))
    )).tracedFit(rawData.withColumn("label", logistic('label)))
  }
}
