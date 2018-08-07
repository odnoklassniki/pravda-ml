package org.apache.spark.ml.odkl

import java.io.File

import odkl.analysis.spark.TestEnv
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.linalg.{BLAS, Vectors}
import org.apache.spark.ml.odkl.ModelWithSummary.Block
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
  with HasLossHistory with HasFeaturesSignificance {
  @transient lazy val model: XGBoostModel = XGBoostSpec._model

  @transient lazy val data: DataFrame = XGBoostSpec._treeData

  lazy val reReadModel: XGBoostModel = XGBoostSpec._reReadModel

  lazy val pipelineModel: PipelineModel = XGBoostSpec._pipelineModel

  "XGBoost " should " train  a model" in {

    model shouldNot be(null)
  }

  "XGBoost " should " predict classes" in {

    val auc = new BinaryClassificationMetrics(
      model.transform(data)
        .select(model.dlmc.getPredictionCol, model.dlmc.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    auc should be > 0.93
  }

  "XGBoost " should " predict classes after re-read" in {
    val auc = new BinaryClassificationMetrics(
      reReadModel.transform(data)
        .select(reReadModel.dlmc.getPredictionCol, reReadModel.dlmc.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    auc should be > 0.93
  }

  "XGBoost " should " survive pipeline round-trip" in {
    val auc = new BinaryClassificationMetrics(
      pipelineModel.transform(data)
        .select(model.dlmc.getPredictionCol, model.dlmc.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    auc should be > 0.93
  }

  "XGBoost " should " predict classes for new model" in {
    val model = XGBoostSpec.createEstimator().fit(_noInterceptDataLogistic)


    val auc = new BinaryClassificationMetrics(
      model.transform(_noInterceptDataLogistic)
        .select(model.dlmc.getPredictionCol, model.dlmc.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    auc should be > 0.93
  }

  "XGBoost " should " add loss history" in {

    val loss = model.summary(lossHistory)

    loss.count() should be > 0L
    loss.schema.size should be(2)
  }

  "XGBoost " should " add loss history with test" in {

    val loss = XGBoostSpec.createEstimator().setTrainTestRation(0.2).fit(XGBoostSpec._treeData).summary(lossHistory)

    loss.count() should be > 0L
    loss.schema.size should be(3)

    loss.where("loss > testLoss").count should be(0)
  }
  
  "XGBoost " should " add significance summary block" in {

    val df = model.summary(featuresSignificance)

    df.count() should be(4)
    df.schema.size should be(3)

    df.where("name = 'first'").select(significance).first().getDouble(0) should be > 0.0
    df.where("name = 'second'").select(significance).first().getDouble(0) should be > 0.0
    df.where("name = 'flag'").select(significance).first().getDouble(0) should be > 0.0
    df.where("name = 'int'").select(significance).first().getDouble(0) should be > 0.0
  }

  "XGBoost " should " add raw trees summary block" in {

    val df = model.summary(Block("rawTrees"))

    df.count() should be(15)
    df.schema.size should be(2)
  }
}

object XGBoostSpec extends WithTestData {

  import sqlc.implicits._

  @transient lazy val _rawTreeData: DataFrame =
    sc.parallelize(Array.tabulate(10000) { _ => {
      val flag = random.nextBoolean()
      val x = Vectors.dense(
        random.nextDouble() * 2 - 1.0,
        random.nextDouble() * 2 - 1.0)
      val int = random.nextInt(10) - 5

      val label = logit(BLAS.dot(x, hiddenModel) + (if(flag) int else -int) / 10.0)


      (x(0), x(1), flag, int, label)
    }
    },2).toDF("first", "second", "flag", "int", "label")

  @transient lazy val _treeData: DataFrame = new NullToNaNVectorAssembler()
    .setInputCols(Array("first", "second", "flag", "int"))
    .setOutputCol("features").transform(_rawTreeData).cache()


  @transient lazy val _model: XGBoostModel = {

    val estimator: XGBoostEstimator = createEstimator()

    estimator.fit(_treeData)
  }

  private def createEstimator(): XGBoostEstimator = {
    new XGBoostEstimator()
      .setLambda(0.0)
      .setObjective("binary:logistic")
      .setEta(0.01f)
      .setMaxDepth(7)
      .setNumRounds(15)
  }

  @transient lazy val _pipelineModel: PipelineModel = {

    val estimator: XGBoostEstimator = createEstimator()

    roundTrip(roundTrip(new Pipeline().setStages(Array(estimator)), Pipeline).fit(_treeData), PipelineModel)
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
