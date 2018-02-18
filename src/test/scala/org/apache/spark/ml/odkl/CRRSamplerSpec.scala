package org.apache.spark.ml.odkl

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions
import org.scalatest.FlatSpec



class CRRSamplerSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations with WithTestData {

  lazy val pureRegressionModel = UnwrappedStage.dataOnly(
    new LogisticDSVRGD(),
    new CRRSamplerModel().setRankingPower(0.0)
  ).fit(noInterceptDataLogistic)

  lazy val pureRankingModel = UnwrappedStage.dataOnly(
    new LogisticDSVRGD(),
    new CRRSamplerModel().setRankingPower(1.0)
  ).fit(noInterceptDataLogistic)

  lazy val combinedSampledModel = UnwrappedStage.dataOnly(
    new LogisticDSVRGD(),
    new CRRSamplerModel().setRankingPower(0.5).setItemSampleRate(0.25)
  ).fit(noInterceptDataLogistic)


  lazy val combinedGroupedSampledModel = UnwrappedStage.dataOnly(
    new LogisticDSVRGD(),
    new CRRSamplerModel().setRankingPower(0.5).setItemSampleRate(0.25).setGroupByColumns("group")
  ).fit(
    noInterceptDataLogistic
      .withColumn("group", functions.expr("FLOOR(RAND() * 10)"))
      .repartition(3, functions.expr("group"))
      .sortWithinPartitions("group"))

  lazy val combinedGroupedSampledModelDynamic = UnwrappedStage.dataOnlyWithTraining(
    new LogisticDSVRGD(),
    new CRRSamplerEstimator().setRankingPower(0.5).setExpectedNumberOfSamples(250).setGroupByColumns("group")
  ).fit(
    noInterceptDataLogistic
      .withColumn("group", functions.expr("FLOOR(RAND() * 10)"))
      .repartition(3, functions.expr("group"))
      .sortWithinPartitions("group"))


  "Regression " should " predict classes" in {
    val model = pureRegressionModel

    val auc = new BinaryClassificationMetrics(
      model.transform(noInterceptDataLogistic)
        .select(model.getPredictionCol, model.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    auc should be >= 0.99
  }

  "Ranking " should " predict classes" in {
    val model = pureRankingModel

    val auc = new BinaryClassificationMetrics(
      model.transform(noInterceptDataLogistic)
        .select(model.getPredictionCol, model.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    auc should be >= 0.99
  }

  "Combined mode " should " predict classes" in {
    val model = combinedSampledModel

    val auc = new BinaryClassificationMetrics(
      model.transform(noInterceptDataLogistic)
        .select(model.getPredictionCol, model.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    auc should be >= 0.99
  }

  "Combined grouped mode " should " predict classes" in {
    val model = combinedGroupedSampledModel

    val auc = new BinaryClassificationMetrics(
      model.transform(noInterceptDataLogistic)
        .select(model.getPredictionCol, model.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    auc should be >= 0.99
  }

  "Dynamically created model " should " predict classes" in {
    val model = combinedGroupedSampledModelDynamic

    val auc = new BinaryClassificationMetrics(
      model.transform(noInterceptDataLogistic)
        .select(model.getPredictionCol, model.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    auc should be >= 0.99
  }
}
