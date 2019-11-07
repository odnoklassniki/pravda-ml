package org.apache.spark.ml.odkl

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.odkl.CrossValidator.FoldsAssigner
import org.apache.spark.sql.{DataFrame, functions}
import org.scalatest.FlatSpec

/**
  * Created by dmitriybugaichenko on 25.01.16.
  */
class EvaluationsSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations with WithModels with HasMetricsBlock {

  
  lazy val noInterceptBinaryPredictions = noInterceptLogisticModel.transform(noInterceptDataLogistic)

  lazy val directBinaryMetrics = new BinaryClassificationMetrics(
    noInterceptBinaryPredictions
      .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol).rdd
      .map(r => (r.getDouble(0), r.getDouble(1))),
    numBins = 100)

  lazy val evaluatedBinaryMetrics = new BinaryClassificationEvaluator().transform(noInterceptBinaryPredictions).collect()

  lazy val evaluatedBinaryMetricsWithFMeasureThresholds = new BinaryClassificationEvaluator().setFmeasureThresholds(Map("f1" -> 1.0, "f01" -> 0.1))
    .transform(noInterceptBinaryPredictions).collect()

  lazy val evaluatedBinaryMetricsMap = evaluatedBinaryMetrics.map(x => x.getString(0) -> x.getDouble(1)).toMap

  lazy val modelWithMetrics: LogisticRegressionModel = {
    val estimator = Evaluator.evaluate(new LogisticRegressionLBFSG(), new BinaryClassificationEvaluator())
    estimator.fit(noInterceptDataLogistic)
  }

  lazy val crossValidationModel = {
    val estimator = Evaluator.crossValidate(
      new LogisticRegressionLBFSG(),
      new TrainTestEvaluator(new BinaryClassificationEvaluator()),
      numFolds = 2,
      numThreads = 1)
    estimator.fit(noInterceptDataLogistic)
  }

  lazy val timeBasedValidatedModel = {
    val estimator = Evaluator.addFolds(
      estimator = Evaluator.validateInFolds(
      new LogisticRegressionLBFSG(),
      new TrainTestEvaluator(new BinaryClassificationEvaluator()),
      numFolds = 2,
      numThreads = 1)
      .setTestSetExpression("IF(time > 0.8, true, IF(foldNum != __FOLD__, false, cast(null as boolean)))"),
      folder = new FoldsAssigner().setNumFolds(2))

    estimator.fit(noInterceptDataLogistic.withColumn("time", functions.rand(0xdeadbeaf)))
  }

  "Binary evaluator " should " should produce same AUC" in {
    evaluatedBinaryMetricsMap("auc") should be(directBinaryMetrics.areaUnderROC() +- delta)
  }

  "Binary evaluator " should " should produce same AUC_PR" in {
    evaluatedBinaryMetricsMap("auc_pr") should be(directBinaryMetrics.areaUnderPR() +- delta)
  }

  "Binary evaluator " should " should produce same ROC" in {
    val evaluated = evaluatedBinaryMetrics.filter(x => x.getString(0).equals("tp_rate")).map(x => x.getDouble(3) -> x.getDouble(1))

    val direct: Array[(Double, Double)] = directBinaryMetrics.roc().collect()

    direct.zip(evaluated).foreach(pair => {
      pair._2._1 should be(pair._1._1)
      pair._2._2 should be(pair._1._2 +- delta)
    })
  }

  "Binary evaluator " should " should produce same PR" in {
    val evaluated = evaluatedBinaryMetrics
      .filter(x => x.getString(0).equals("precision") && "recall".equals(x.getString(2)))
      .map(x => x.getDouble(3) -> x.getDouble(1))

    val direct: Array[(Double, Double)] = directBinaryMetrics.pr().collect()

    direct.zip(evaluated).foreach(pair => {
      pair._2._1 should be(pair._1._1)
      pair._2._2 should be(pair._1._2 +- delta)
    })
  }

  "Binary evaluator " should " should produce same precision by threshold" in {
    val evaluated = evaluatedBinaryMetrics
      .filter(x => x.getString(0).equals("precision") && "threshold".equals(x.getString(2)))
      .map(x => x.getDouble(3) -> x.getDouble(1))

    val direct: Array[(Double, Double)] = directBinaryMetrics.precisionByThreshold().collect()

    direct.zip(evaluated).foreach(pair => {
      pair._2._1 should be(pair._1._1)
      pair._2._2 should be(pair._1._2 +- delta)
    })
  }

  "Binary evaluator " should " should produce same recall by threshold" in {
    val evaluated = evaluatedBinaryMetrics
      .filter(x => x.getString(0).equals("recall") && "threshold".equals(x.getString(2)))
      .map(x => x.getDouble(3) -> x.getDouble(1))

    val direct: Array[(Double, Double)] = directBinaryMetrics.recallByThreshold().collect()

    direct.zip(evaluated).foreach(pair => {
      pair._2._1 should be(pair._1._1)
      pair._2._2 should be(pair._1._2 +- delta)
    })
  }

  "Binary evaluator " should " should produce same f1 by threshold" in {
    val evaluated = evaluatedBinaryMetrics
      .filter(x => x.getString(0).equals("f1") && "threshold".equals(x.getString(2)))
      .map(x => x.getDouble(3) -> x.getDouble(1))

    val direct: Array[(Double, Double)] = directBinaryMetrics.fMeasureByThreshold().collect()

    direct.zip(evaluated).foreach(pair => {
      pair._2._1 should be(pair._1._1)
      pair._2._2 should be(pair._1._2 +- delta)
    })
  }


  "Binary evaluator " should " should produce same f1 and f0.1 by thresholds" in {
    val evaluatedF1 = evaluatedBinaryMetricsWithFMeasureThresholds
      .filter(x => x.getString(0).equals("f1") && "threshold".equals(x.getString(2)))
      .map(x => x.getDouble(3) -> x.getDouble(1))

    val directF1: Array[(Double, Double)] = directBinaryMetrics.fMeasureByThreshold().collect()

    val evaluatedF01 = evaluatedBinaryMetricsWithFMeasureThresholds
      .filter(x => x.getString(0).equals("f01") && "threshold".equals(x.getString(2)))
      .map(x => x.getDouble(3) -> x.getDouble(1))

    val directF01: Array[(Double, Double)] = directBinaryMetrics.fMeasureByThreshold(0.1).collect()

    directF1.zip(evaluatedF1).foreach(pair => {
      pair._2._1 should be(pair._1._1)
      pair._2._2 should be(pair._1._2 +- delta)
    })

    directF01.zip(evaluatedF01).foreach(pair => {
      pair._2._1 should be(pair._1._1)
      pair._2._2 should be(pair._1._2 +- delta)
    })
  }


  "Evaluating estimator " should " add metrics to summary" in {
    modelWithMetrics.summary.blocks(metrics).collect().zip(evaluatedBinaryMetrics).foreach(pair => {
      pair._1.getString(0) should be(pair._2.getString(0))
      pair._1.getString(2) should be(pair._2.getString(2))

      Math.abs(pair._1.getDouble(1) - pair._2.getDouble(1)) should be <= delta
      if (pair._1.isNullAt(3)) {
        pair._2.isNullAt(3) should be(true)
      } else {
        pair._1.getDouble(3) should be(pair._2.getDouble(3) +- delta)
      }
    })
  }

  "Evaluating estimator " should " not miss weights" in {
    val model = modelWithMetrics
    val summary = model.summary

    val weigths = (summary $ model.weights).rdd.map(r => r.getInt(0) -> r.getDouble(2)).collect().toMap

    weigths(0) should be(model.getCoefficients(0))
    weigths(1) should be(model.getCoefficients(1))
  }

  "Evaluating estimator " should " not set coefficients weights" in {
    val model = modelWithMetrics

    model.getCoefficients(0) should be(noInterceptLogisticModel.getCoefficients(0) +- 0.000001)
    model.getCoefficients(1) should be(noInterceptLogisticModel.getCoefficients(1) +- 0.000001)
    model.getIntercept should be(noInterceptLogisticModel.getIntercept +- 0.000001)
  }

  "Cross validation " should " add total metrics to summary" in {
    val metrics: DataFrame = crossValidationModel.summary.blocks(this.metrics)


    metrics.filter(metrics("foldNum") === -1 && metrics("isTest") === false)
      .select("metric", "value", "x-metric", "x-value").collect().zip(evaluatedBinaryMetrics).foreach(pair => {
      pair._1.getString(0) should be(pair._2.getString(0))
      pair._1.getString(2) should be(pair._2.getString(2))

      Math.abs(pair._1.getDouble(1) - pair._2.getDouble(1)) should be <= delta
      if (pair._1.isNullAt(3)) {
        pair._2.isNullAt(3) should be(true)
      } else {
        pair._1.getDouble(3) should be(pair._2.getDouble(3) +- delta)
      }
    })
  }

  "Cross validation " should " add expected number of folds to metircs" in { 
    val metrics: DataFrame = crossValidationModel.summary.blocks(this.metrics)

    val configs = metrics.select("foldNum", "isTest").distinct().collect().map(x => x.getInt(0) -> x.getBoolean(1))

    configs.sorted should be(Seq(
      -1 -> false,
      0 -> false,
      0 -> true,
      1 -> false,
      1 -> true
    ).sorted)
  }

  "Cross validation " should " add weights for main fold" in {
    val model = crossValidationModel
    val summary = model.summary

    val weightsFrame: DataFrame = summary $ model.weights
    val weigths = weightsFrame
      .filter(weightsFrame("foldNum") === -1).rdd
      .map(r => r.getInt(0) -> r.getDouble(2)).collect().toMap

    weigths(0) should be(model.getCoefficients(0))
    weigths(1) should be(model.getCoefficients(1))
  }

  "Cross validation " should " produce similar weights for folds" in {
    val model = crossValidationModel
    val summary = model.summary

    val weightsFrame: DataFrame = summary $ model.weights

    for(i <- -1 to 1) {
      val weigths = weightsFrame
        .filter(weightsFrame("foldNum") === 0)
        .select("index", "weight").rdd
        .map(r => r.getInt(0) -> r.getDouble(1)).collect().toMap

      val average = Vectors.dense(Array(weigths(0), weigths(1)))

      cosineDistance(average, model.getCoefficients) should be <= 0.00001
    }
  }

  "Cross validation " should " add expected number of folds to weights" in {
    val model = crossValidationModel
    val summary = model.summary

    val weightsFrame: DataFrame = summary $ model.weights
    val configs = weightsFrame.select("foldNum").distinct().collect().map(x => x.getInt(0))

    configs.sorted should be(Seq(-1, 0, 1).sorted)
  }

  "Cross validation " should " not miss coefficients weights" in {
    val model = crossValidationModel

    model.getCoefficients(0) should be(noInterceptLogisticModel.getCoefficients(0) +- 0.000001)
    model.getCoefficients(1) should be(noInterceptLogisticModel.getCoefficients(1) +- 0.000001)
    model.getIntercept should be(noInterceptLogisticModel.getIntercept +- 0.000001)
  }

  "Time based cross validation " should " add total metrics to summary" in {
    val metrics: DataFrame = timeBasedValidatedModel.summary.blocks(this.metrics)


    metrics.filter(metrics("foldNum") === -1 && metrics("isTest") === false)
      .select("metric", "value", "x-metric", "x-value").collect().zip(evaluatedBinaryMetrics).foreach(pair => {
      pair._1.getString(0) should be(pair._2.getString(0))
      pair._1.getString(2) should be(pair._2.getString(2))

      Math.abs(pair._1.getDouble(1) - pair._2.getDouble(1)) should be <= delta
      if (pair._1.isNullAt(3)) {
        pair._2.isNullAt(3) should be(true)
      } else {
        pair._1.getDouble(3) should be(pair._2.getDouble(3) +- delta)
      }
    })
  }

  "Time based cross validation " should " add expected number of folds to metircs" in {
    val metrics: DataFrame = timeBasedValidatedModel.summary.blocks(this.metrics)

    val configs = metrics.select("foldNum", "isTest").distinct().collect().map(x => x.getInt(0) -> x.getBoolean(1))

    configs.sorted should be(Seq(
      -1 -> false,
      0 -> false,
      0 -> true,
      1 -> false,
      1 -> true
    ).sorted)
  }

  "Time based cross validation " should " add weights for main fold" in {
    val model = timeBasedValidatedModel
    val summary = model.summary

    val weightsFrame: DataFrame = summary $ model.weights
    val weigths = weightsFrame
      .filter(weightsFrame("foldNum") === -1).rdd
      .map(r => r.getInt(0) -> r.getDouble(2)).collect().toMap

    weigths(0) should be(model.getCoefficients(0))
    weigths(1) should be(model.getCoefficients(1))
  }

  "Time based cross validation " should " produce similar weights for folds" in {
    val model = timeBasedValidatedModel
    val summary = model.summary

    val weightsFrame: DataFrame = summary $ model.weights

    for(i <- -1 to 1) {
      val weigths = weightsFrame
        .filter(weightsFrame("foldNum") === 0)
        .select("index", "weight").rdd
        .map(r => r.getInt(0) -> r.getDouble(1)).collect().toMap

      val average = Vectors.dense(Array(weigths(0), weigths(1)))

      // Smaller folds, smaller limits
      cosineDistance(average, model.getCoefficients) should be <= 0.0001
    }
  }

  "Time based cross validation " should " add expected number of folds to weights" in {
    val model = timeBasedValidatedModel
    val summary = model.summary

    val weightsFrame: DataFrame = summary $ model.weights
    val configs = weightsFrame.select("foldNum").distinct().collect().map(x => x.getInt(0))

    configs.sorted should be(Seq(-1, 0, 1).sorted)
  }

  "Time based cross validation " should " not miss coefficients weights" in {
    val model = timeBasedValidatedModel

    model.getCoefficients(0) should be(noInterceptLogisticModel.getCoefficients(0) +- 0.000001)
    model.getCoefficients(1) should be(noInterceptLogisticModel.getCoefficients(1) +- 0.000001)
    model.getIntercept should be(noInterceptLogisticModel.getIntercept +- 0.000001)
  }

}
