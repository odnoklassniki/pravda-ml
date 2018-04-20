package org.apache.spark.ml.odkl

import java.io.File

import odkl.analysis.spark.TestEnv
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.odkl.WithTestData._
import org.apache.spark.ml.util.{Identifiable, MLReadable, MLWritable}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame
import org.scalatest.FlatSpec


/**
  * Created by vyacheslav.baranov on 02/12/15.
  */
class XGBoostSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with WithTestData
  with HasLossHistory {
  @transient lazy val model: XGBoostModel = XGBoostSpec._model

  @transient lazy val data: DataFrame = noInterceptDataLogistic

  lazy val reReadModel: XGBoostModel = XGBoostSpec._reReadModel

  lazy val pipelineModel: PipelineModel = XGBoostSpec._pipelineModel

  "XGBoost " should " train a model" in {

    model shouldNot be(null)
  }

  "XGBoost " should " predict classes" in {

    val auc = new BinaryClassificationMetrics(
      model.transform(data)
        .select(model.dlmc.getPredictionCol, model.dlmc.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    auc should be > 0.96
  }

  "XGBoost " should " predict classes after re-read" in {
    val auc = new BinaryClassificationMetrics(
      reReadModel.transform(data)
        .select(reReadModel.dlmc.getPredictionCol, reReadModel.dlmc.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    auc should be > 0.96
  }

  "XGBoost " should " survive pipeline round-trip" in {
    val auc = new BinaryClassificationMetrics(
      pipelineModel.transform(data)
        .select(model.dlmc.getPredictionCol, model.dlmc.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    auc should be > 0.96
  }

  "XGBoost " should " add loss history" in {

    val loss = model.summary(lossHistory)

    loss.count() should be > 0L
    loss.schema.size should be(2)
  }

  "XGBoost " should " add loss history with test" in {

    val loss = XGBoostSpec.createEstimator().setTrainTestRation(0.2).fit(noInterceptDataLogistic).summary(lossHistory)

    loss.count() should be > 0L
    loss.schema.size should be(3)

    loss.where("loss > testLoss").count should be(0)
  }

  "XGBoost " should " predict better with more depth" in {
    val model = XGBoostSpec.createEstimator().setMaxDepth(5).fit(noInterceptDataLogistic)

    val auc = new BinaryClassificationMetrics(
      model.transform(data)
        .select(model.dlmc.getPredictionCol, model.dlmc.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    auc should be > 0.99
  }
}

object XGBoostSpec extends WithTestData {

  @transient lazy val _model: XGBoostModel = {

    val estimator: XGBoostEstimator = createEstimator()

    estimator.fit(_noInterceptDataLogistic)
  }

  private def createEstimator(): XGBoostEstimator = {
    new XGBoostEstimator()
      .setObjective("binary:logistic")
      .setEta(0.01f)
      .setMaxDepth(3)
  }

  @transient lazy val _pipelineModel: PipelineModel = {

    val estimator: XGBoostEstimator = createEstimator()

    roundTrip(new Pipeline().setStages(Array(estimator)).fit(_noInterceptDataLogistic), PipelineModel)
  }

  @transient lazy val _reReadModel: XGBoostModel = roundTrip(_model, XGBoostModel)

  def roundTrip[M <: MLWritable with Identifiable](data: M, reader: MLReadable[M]): M = {
    val directory = new File(FileUtils.getTempDirectory, data.uid)
    try {
      data.save(directory.getAbsolutePath)

      val reReadModel = reader.load(directory.getAbsolutePath)

      reReadModel
    } finally {
      FileUtils.deleteDirectory(directory)
    }
  }
}
