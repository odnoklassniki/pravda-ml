package org.apache.spark.ml.odkl

import java.io.File
import java.util.concurrent.{CountDownLatch, ThreadLocalRandom, TimeUnit}

import breeze.linalg
import breeze.numerics.abs
import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.scheduler.{SparkListener, SparkListenerUnpersistRDD}
import org.apache.spark.sql.{DataFrame, functions}
import org.mockito.internal.verification.Description
import org.mockito.invocation.InvocationOnMock
import org.mockito.stubbing.Answer
import org.mockito.{ArgumentMatcher, ArgumentMatchers, Matchers, Mockito}
import org.scalatest.FlatSpec


/**
  * Created by vyacheslav.baranov on 02/12/15.
  */
class LinearModelsSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations with WithModels {

  "SGD " should " train a model" in {

    val estimator = new LinearRegressionSGD()

    val model = estimator.fit(noInterceptData)

    val dev: linalg.Vector[Double] = hiddenModel.toBreeze - model.getCoefficients.toBreeze

    val deviation: Double = dev dot dev

    deviation should be <= delta
    model.getIntercept should be(0.0)
  }

  "SGD " should " train intercept" in {

    val model = interceptedSgdModel

    val dev: linalg.Vector[Double] = hiddenModel.toBreeze - model.getCoefficients.toBreeze

    val deviation: Double = dev dot dev

    deviation should be <= delta
    abs(model.getIntercept - hiddenIntercept) should be <= delta
  }

  "LBFSG " should " train a model" in {
    val model = noInterceptLogisticModel

    val deviation: Double = cosineDistance(hiddenModel, model.getCoefficients)

    abs(deviation) should be <= delta
    model.getIntercept should be(0.0)
  }

  "LBFSG " should " train intercept" in {
    val model = interceptLogisticModel

    val deviation: Double = cosineDistance(
      Vectors.dense(this.hiddenModel.toArray :+ hiddenIntercept),
      Vectors.dense(model.getCoefficients.toArray :+ model.getIntercept))

    abs(deviation) should be <= delta
  }

  "Model " should " predict linearly" in {

    val coefficients = interceptedSgdModel.getCoefficients
    val intercept = interceptedSgdModel.getIntercept

    val rmse = Math.sqrt(interceptedSgdModel.transform(interceptData)
      .select(interceptedSgdModel.getFeaturesCol, interceptedSgdModel.getPredictionCol)
      .map(r => {
        val expectedPrediction: Double = BLAS.dot(r.getAs[Vector](0), coefficients) + intercept
        (expectedPrediction - r.getDouble(1)) * (expectedPrediction - r.getDouble(1))
      })
      .sum())


    rmse should be(0.0)
  }

  "Model " should " predict labels" in {

    val rmse = Math.sqrt(interceptedSgdModel.transform(interceptData)
      .select(interceptedSgdModel.getLabelCol, interceptedSgdModel.getPredictionCol)
      .map(r => (r.getDouble(0) - r.getDouble(1)) * (r.getDouble(0) - r.getDouble(1)))
      .mean())


    rmse should be <= 0.1
  }

  "Model " should " predict classes" in {
    val model = noInterceptLogisticModel

    val auc = new BinaryClassificationMetrics(
      model.transform(noInterceptDataLogistic)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    auc should be >= 0.9
  }

  "Regularized model " should " predict classes" in {
    val model = noInterceptLogisticRegularizedModel

    val auc = new BinaryClassificationMetrics(
      model.transform(noInterceptDataLogistic)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    auc should be >= 0.9
  }

  "Regularized model " should " produce lower weights" in {
    val reg = noInterceptLogisticRegularizedModel.getCoefficients.toArray.map(x => x * x).sum
    val noReg = noInterceptLogisticModel.getCoefficients.toArray.map(x => x * x).sum

    noReg should be > reg
  }

  "Model " should " predict better with intercept" in {
    val model = noInterceptLogisticModel
    val modelIntercepted = interceptLogisticModel

    val auc = new BinaryClassificationMetrics(
      model.transform(interceptDataLogistig)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    val aucIntercepted = new BinaryClassificationMetrics(
      modelIntercepted.transform(interceptDataLogistig)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    aucIntercepted should be >= 0.9
    aucIntercepted should be > auc
  }

  "Model " should " predict better with scaler" in {

    val estimator = new LogisticRegressionLBFSG()

    val scale = functions.udf[Vector, Vector](x => {
      Vectors.dense(x(0) * 5 - 10, x(1) * 100000 + 20000)
    })

    val scaled = noInterceptDataLogistic.withColumn("features", scale(noInterceptDataLogistic("features")))

    val model = estimator.fit(scaled)
    val modelScaled = Scaler.scale(estimator).fit(scaled)

    val auc = new BinaryClassificationMetrics(
      model.transform(scaled)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    val aucScaled = new BinaryClassificationMetrics(
      modelScaled.transform(scaled)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    aucScaled should be >= 0.9
    aucScaled should be > auc
  }

  "Model " should " predict better with scaler for sparce vectors" in {

    val estimator = new LogisticRegressionLBFSG()

    val scale = functions.udf[Vector, Vector](x => {
      Vectors.sparse(2, Array(0, 1), Array(x(0) * 5 - 10, x(1) * 100000 + 20000))
    })

    val scaled = noInterceptDataLogistic.withColumn("features", scale(noInterceptDataLogistic("features")))

    val model = estimator.fit(scaled)
    val modelScaled = Scaler.scale(estimator).fit(scaled)

    val auc = new BinaryClassificationMetrics(
      model.transform(scaled)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    val aucScaled = new BinaryClassificationMetrics(
      modelScaled.transform(scaled)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    aucScaled should be >= 0.9
    aucScaled should be > auc
  }

  lazy val scaledData = {
    val scale = functions.udf[Vector, Vector](x => {
      Vectors.dense(x(0) * 5 - 10, x(1) * 100000 + 20000)
    })

    interceptDataLogistig.withColumn(
      "features",
      scale(interceptDataLogistig("features")).as("features", interceptDataLogistig.schema("features").metadata))
  }

  lazy val scaledModel = Scaler.scale(Interceptor.intercept(new LogisticRegressionLBFSG())).fit(scaledData)

  "Model " should " predict better with scaler and intercept" in {

    val model = Interceptor.intercept(new LogisticRegressionLBFSG()).fit(scaledData)

    val auc = new BinaryClassificationMetrics(
      model.transform(scaledData)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    val aucScaled = new BinaryClassificationMetrics(
      scaledModel.transform(scaledData)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    aucScaled should be >= 0.9
    aucScaled should be > auc
  }

  lazy val reReadModel = {
    val directory = new File(FileUtils.getTempDirectory, interceptedSgdModel.uid)
    try {
      interceptedSgdModel.save(directory.getAbsolutePath)

      val reReadModel = ModelWithSummary.reader(classOf[LinearRegressionModel]).context(sqlc).load(directory.getAbsolutePath)

      reReadModel.summary.blocks.foreach(x => x._2.cache().count())

      reReadModel
    } finally {
      FileUtils.deleteDirectory(directory)
    }
  }

  "Model " should " save parameters" in {
    reReadModel.getIntercept should be(interceptedSgdModel.getIntercept)
    reReadModel.getCoefficients should be(interceptedSgdModel.getCoefficients)
  }

  "Model " should " save summary" in {
    val summary = reReadModel.summary

    val weigths = (summary $ reReadModel.weights).map(r => r.getInt(0) -> r.getDouble(2)).collect().toMap

    weigths(0) should be(reReadModel.getCoefficients(0))
    weigths(1) should be(reReadModel.getCoefficients(1))
    weigths(-1) should be(reReadModel.getIntercept)
  }

  "Model " should " be able to predict labels after read" in {
    val rmse = Math.sqrt(reReadModel.transform(interceptData)
      .select(reReadModel.getLabelCol, reReadModel.getPredictionCol)
      .map(r => (r.getDouble(0) - r.getDouble(1)) * (r.getDouble(0) - r.getDouble(1)))
      .mean())


    rmse should be <= 0.1
  }

  "Model " should " be able to predict classes after read" in {
    val original = noInterceptLogisticModel
    val model = {
      val directory = new File(FileUtils.getTempDirectory, original.uid)
      try {
        original.save(directory.getAbsolutePath)

        ModelWithSummary.reader(classOf[LogisticRegressionModel]).context(sqlc).load(directory.getAbsolutePath)
      } finally {
        FileUtils.deleteDirectory(directory)
      }
    }

    val auc = new BinaryClassificationMetrics(
      model.transform(noInterceptDataLogistic)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol)
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    auc should be >= 0.9
  }



  "Regressor " should " set column names in   summary" in {

    val summary = interceptedSgdModel.summary

    val names = (summary $ interceptedSgdModel.weights).map(r => r.getInt(0) -> r.getString(1)).collect().toMap

    names(0) should be("first")
    names(1) should be("second")
    names(-1) should be("#intercept")
  }

  "Regressor " should " set column weights in   summary" in {

    val model = interceptedSgdModel
    val summary = model.summary

    val weigths = (summary $ model.weights).map(r => r.getInt(0) -> r.getDouble(2)).collect().toMap

    weigths(0) should be(model.getCoefficients(0))
    weigths(1) should be(model.getCoefficients(1))
    weigths(-1) should be(model.getIntercept)
  }

  "Regressor " should " set unscaled column weights and names summary" in {

    val model = scaledModel
    val summary = model.summary

    val weightsFrame: DataFrame = summary $ model.weights
    val weightIndex = weightsFrame.schema.fieldIndex("weight")
    val weigths = weightsFrame.map(r => r.getInt(0) -> r.getDouble(weightIndex)).collect().toMap

    weigths(0) should be(model.getCoefficients(0))
    weigths(1) should be(model.getCoefficients(1))
    weigths(-1) should be(model.getIntercept)

    val names = weightsFrame.map(r => r.getInt(0) -> r.getString(1)).collect().toMap

    names(0) should be("first")
    names(1) should be("second")
    names(-1) should be("#intercept")
  }

  "Regressor " should " cache data by default" in {

    val estimator = new LinearRegressionSGD()

    val id = sc.parallelize(Seq("Some")).id

    val listener: SparkListener = Mockito.mock(classOf[SparkListener])
    sc.addSparkListener(listener)

    val countDown = createWaiter(listener)(_.onUnpersistRDD(ArgumentMatchers.argThat(rddOlderThan(id))))

    estimator.fit(interceptData)

    countDown.await(1000, TimeUnit.MILLISECONDS) should be(true)

    Mockito.verify(listener, Mockito.times(1)).onUnpersistRDD(ArgumentMatchers.argThat(rddOlderThan(id)))
  }

  "Regressor " should " not cache data if disabled" in {

    val estimator = new LinearRegressionSGD()
    estimator.set(estimator.cacheTrainData, false)

    val id = sc.parallelize(Seq("Some")).id

    val listener: SparkListener = Mockito.mock(classOf[SparkListener])
    sc.addSparkListener(listener)

    val countDown = createWaiter(listener)(_.onUnpersistRDD(ArgumentMatchers.argThat(rddOlderThan(id))))

    estimator.fit(interceptData)

    countDown.await(1000, TimeUnit.MILLISECONDS) should be(false)

    Mockito.verify(listener, Mockito.times(0)).onUnpersistRDD(ArgumentMatchers.argThat(rddOlderThan(id)))
  }

  def rddOlderThan(id: Int): ArgumentMatcher[SparkListenerUnpersistRDD] = {
    new ArgumentMatcher[SparkListenerUnpersistRDD]() {
      override def matches(item: SparkListenerUnpersistRDD): Boolean = {
        item.rddId > id
      }
    }
  }

  def createWaiter[T](mock: T, count: Int = 1)(call: T => Any): CountDownLatch = {

    val countDown = new CountDownLatch(count)

    call(Mockito.doAnswer(new Answer[Unit] {
      override def answer(invocation: InvocationOnMock): Unit = {
        countDown.countDown()
      }
    }).when(mock))

    countDown
  }

  "Regularization " should " eliminate irrelevant" in {

    val addRandom = functions.udf[Vector,Vector](x => {
      Vectors.dense(2 * ThreadLocalRandom.current().nextDouble() - 1.0, x.toArray :_*)
    })

    val withIrrelevant = noInterceptDataLogistic.withColumn(
      "features",
      addRandom(noInterceptDataLogistic("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val weights = new LogisticRegressionLBFSG().setRegParam(0.035).setElasticNetParam(1.0)
      .fit(withIrrelevant).getCoefficients

    Math.abs(weights(0)) should be <= 0.0
    Math.abs(weights(1)) should be > 0.0
    Math.abs(weights(2)) should be > 0.0

    cosineDistance(Vectors.dense(weights.toArray.drop(1)), hiddenModel) should be <= 0.001
  }

  "Regularization " should " eliminate correlated" in {

    val noise = 0.05

    val addCorrelated = functions.udf[Vector,Vector](x => {
      Vectors.dense(x.toArray(0) / 1.5  +  2 * noise * ThreadLocalRandom.current().nextDouble() - noise, x.toArray :_*)
    })

    val withCorrelated = noInterceptDataLogistic.withColumn(
      "features",
      addCorrelated(noInterceptDataLogistic("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val weights = new LogisticRegressionLBFSG().setRegParam(0.03).setElasticNetParam(1.0)
      .fit(withCorrelated).getCoefficients

    Math.abs(weights(0)) should be <= 0.0
    Math.abs(weights(1)) should be > 0.0
    Math.abs(weights(2)) should be > 0.0

    cosineDistance(Vectors.dense(weights.toArray.drop(1)), hiddenModel) should be <= 0.001
  }

  "Regularization " should " remove intercept by default" in {
    val model = Interceptor.intercept(new LogisticRegressionLBFSG().setRegParam(0.03).setElasticNetParam(1.0).setPreInitIntercept(true))
      .fit(noInterceptDataLogistic)

    val weights = model.getCoefficients
    val intercept = model.getIntercept

    Math.abs(intercept) should be <= 0.0
    Math.abs(weights(0)) should be > 0.0
    Math.abs(weights(1)) should be > 0.0

    cosineDistance(weights, hiddenModel) should be <= 0.001
  }

  "Regularization " should " keep intercept if asked" in {

    val model = Interceptor.intercept(new LogisticRegressionLBFSG().setRegParam(0.03).setElasticNetParam(1.0).setRegularizeLast(false))
      .fit(noInterceptDataLogistic)

    val weights = model.getCoefficients
    val intercept = model.getIntercept

    Math.abs(intercept) should be > 0.0
    Math.abs(weights(0)) should be > 0.0
    Math.abs(weights(1)) should be > 0.0

    cosineDistance(weights, hiddenModel) should be <= 0.001
  }

  "Regularization " should " keep intercept if relevant" in {
    val model = Interceptor.intercept(new LogisticRegressionLBFSG().setRegParam(0.03).setElasticNetParam(1.0).setPreInitIntercept(true))
      .fit(interceptDataLogistig)

    val weights = model.getCoefficients
    val intercept = model.getIntercept

    Math.abs(intercept) should be > 0.0
    Math.abs(weights(0)) should be > 0.0
    Math.abs(weights(1)) should be > 0.0

    cosineDistance(weights, hiddenModel) should be <= 0.001
  }

}
