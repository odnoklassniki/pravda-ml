package org.apache.spark.ml.odkl

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg
import breeze.numerics.abs
import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, functions}
import org.scalatest.FlatSpec


class LinearDSVRGDModelsSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations with WithTestData {
  lazy val interceptedSgdModel = LinearDSVRGDModelsSpec._interceptedSgdModel
  lazy val noInterceptLogisticModel = LinearDSVRGDModelsSpec._noInterceptLogisticModel
  lazy val interceptLogisticModel = LinearDSVRGDModelsSpec._interceptLogisticModel


  "DSVRGD " should " train a linear model" in {

    val estimator = new LinearDSVRGD().setLocalMinibatchSize(2)

    val model = estimator.fit(noInterceptData)

    val dev: linalg.Vector[Double] = hiddenModel.asBreeze - model.getCoefficients.asBreeze

    val deviation: Double = dev dot dev

    deviation should be <= delta
    model.getIntercept should be(0.0)
  }

  "DSVRGD " should " train a linear model with intercept" in {

    val model = interceptedSgdModel

    val dev: linalg.Vector[Double] = hiddenModel.asBreeze - model.getCoefficients.asBreeze

    val deviation: Double = dev dot dev

    deviation should be <= delta
    abs(model.getIntercept - hiddenIntercept) should be <= delta
  }

  "DSVRGD " should " train a logistic model" in  {
    val model = noInterceptLogisticModel

    val deviation: Double = cosineDistance(hiddenModel, model.getCoefficients)

    abs(deviation) should be <= delta
    model.getIntercept should be(0.0)
  }

  "DSVRGD " should " train a logistic model with intercept" in {
    val model = interceptLogisticModel

    val deviation: Double = cosineDistance(
      Vectors.dense(this.hiddenModel.toArray :+ hiddenIntercept),
      Vectors.dense(model.getCoefficients.toArray :+ model.getIntercept))

    abs(deviation) should be <= delta
  }

  "Linear model " should " predict labels" in {

    val rmse = Math.sqrt(interceptedSgdModel.transform(interceptData)
      .select(interceptedSgdModel.getLabelCol, interceptedSgdModel.getPredictionCol)
      .rdd
      .map(r => (r.getDouble(0) - r.getDouble(1)) * (r.getDouble(0) - r.getDouble(1)))
      .mean())

    rmse should be <= 0.001
  }

  "Logistic model " should " predict classes" in {
    val model = noInterceptLogisticModel

    val auc = new BinaryClassificationMetrics(
      model.transform(noInterceptDataLogistic)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    auc should be >= 0.999
  }

  "Logistic model " should " predict better with scaler" in {

    val estimator = new LogisticDSVRGD().setLocalMinibatchSize(2)

    val scale = functions.udf[Vector, Vector](x => {
      Vectors.dense(x(0) * 5 - 10, x(1) * 100000 + 20000)
    })

    val scaled = noInterceptDataLogistic.withColumn(
      "features",
      scale(noInterceptDataLogistic("features")).as("features", noInterceptDataLogistic.schema("features").metadata))

    val model = estimator.fit(scaled)
    val modelScaled = Scaler.scale(estimator).fit(scaled)

    val auc = new BinaryClassificationMetrics(
      model.transform(scaled)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    val aucScaled = new BinaryClassificationMetrics(
      modelScaled.transform(scaled)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    aucScaled should be >= 0.999
    aucScaled should be > auc
  }

  "Logistic model " should " predict better with scaler for sparce vectors" in {

    val estimator = new LogisticDSVRGD().setLocalMinibatchSize(2)

    val scale = functions.udf[Vector, Vector](x => {
      Vectors.sparse(2, Array(0, 1), Array(x(0) * 5 - 10, x(1) * 100000 + 20000))
    })

    val scaled = noInterceptDataLogistic.withColumn(
      "features",
      scale(noInterceptDataLogistic("features")).as("features", noInterceptDataLogistic.schema("features").metadata))

    val model = estimator.fit(scaled)
    val modelScaled = Scaler.scale(estimator).fit(scaled)

    val auc = new BinaryClassificationMetrics(
      model.transform(scaled)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    val aucScaled = new BinaryClassificationMetrics(
      modelScaled.transform(scaled)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    aucScaled should be >= 0.999
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

  lazy val scaledModel = Scaler.scale(Interceptor.intercept(new LogisticDSVRGD())).fit(scaledData)

  "Logistic model " should " predict better with scaler and intercept" in {

    val model = Interceptor.intercept(new LogisticDSVRGD().setLocalMinibatchSize(2)).fit(scaledData)

    val auc = new BinaryClassificationMetrics(
      model.transform(scaledData)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    val aucScaled = new BinaryClassificationMetrics(
      scaledModel.transform(scaledData)
        .select(interceptedSgdModel.getPredictionCol, interceptedSgdModel.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    aucScaled should be >= 0.999
    aucScaled should be > auc
  }

  "DSVRGD " should " set column names in summary" in {

    val summary = interceptedSgdModel.summary

    val names = (summary $ interceptedSgdModel.weights).rdd.map(r => r.getInt(0) -> r.getString(1)).collect().toMap

    names(0) should be("first")
    names(1) should be("second")
    names(-1) should be("#intercept")
  }

  "DSVRGD " should " set column weights in summary" in {

    val model = interceptedSgdModel
    val summary = model.summary

    val weigths = (summary $ model.weights).rdd.map(r => r.getInt(0) -> r.getDouble(2)).collect().toMap

    weigths(0) should be(model.getCoefficients(0))
    weigths(1) should be(model.getCoefficients(1))
    weigths(-1) should be(model.getIntercept)
  }

  "DSVRGD " should " set unscaled column weights and names summary" in {

    val model = scaledModel
    val summary = model.summary

    val weightsFrame: DataFrame = summary $ model.weights
    val weightIndex = weightsFrame.schema.fieldIndex("weight")
    val weigths = weightsFrame.rdd.map(r => r.getInt(0) -> r.getDouble(weightIndex)).collect().toMap

    weigths(0) should be(model.getCoefficients(0))
    weigths(1) should be(model.getCoefficients(1))
    weigths(-1) should be(model.getIntercept)

    val names = weightsFrame.rdd.map(r => r.getInt(0) -> r.getString(1)).collect().toMap

    names(0) should be("first")
    names(1) should be("second")
    names(-1) should be("#intercept")
  }

  val hiddenModel2 = Vectors.dense(0.7, -0.1)
  val hiddenModel3 = Vectors.dense(-0.3, 0.4)


  private val calculateLabels: UserDefinedFunction = functions.udf[Vector, Vector](x => Vectors.dense(
    WithTestData.logit(BLAS.dot(x, hiddenModel)),
    WithTestData.logit(BLAS.dot(x, hiddenModel2)),
    WithTestData.logit(BLAS.dot(x, hiddenModel3))
  ))

  lazy val multiClassData = noInterceptData.withColumn(
    "label", calculateLabels(noInterceptData("features")).as("label", new AttributeGroup("label", 3).toMetadata()))

  lazy val weights = Matrices.dense(3, 2, Array(
    hiddenModel(0), hiddenModel2(0), hiddenModel3(0),
    hiddenModel(1), hiddenModel2(1), hiddenModel3(1)
  )).asInstanceOf[DenseMatrix]

  "DSVRGD " should " should produce matrix model" in {

    val model = new LogisticMatrixDSVRGD().setLocalMinibatchSize(2).fit(multiClassData)

    val predictions = model.transform(multiClassData)
    val prediction = predictions(interceptedSgdModel.getPredictionCol)
    val label = predictions(interceptedSgdModel.getLabelCol)

    for (i <- 0 until 3) {
      val extractor = functions.udf[Double, Vector](x => x(i))

      val auc = new BinaryClassificationMetrics(
        predictions
          .select(extractor(prediction), extractor(label)).rdd
          .map(r => (r.getDouble(0), r.getDouble(1))))
        .areaUnderROC()

      auc should be >= 0.99
    }
  }

  "Regularization " should " should produce lower weights with same quality" in {
    val model = new LogisticDSVRGD().setRegParam(0.01).setElasticNetParam(0.0).setLocalMinibatchSize(2)
      .fit(noInterceptDataLogistic)

    val regNorm = Math.sqrt(model.getCoefficients.toArray.map(x=> x*x).sum)
    val noRegNorm = Math.sqrt(noInterceptLogisticModel.getCoefficients.toArray.map(x=> x*x).sum)

    regNorm should be < noRegNorm

    val auc = new BinaryClassificationMetrics(
      model.transform(noInterceptDataLogistic)
        .select(model.getPredictionCol, model.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    val aucNoReg = new BinaryClassificationMetrics(
      noInterceptLogisticModel.transform(noInterceptDataLogistic)
        .select(noInterceptLogisticModel.getPredictionCol, noInterceptLogisticModel.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()

    auc should be >= aucNoReg * 0.99
  }

  "Regularization " should " eliminate irrelevant" in {

    val addRandom = functions.udf[Vector,Vector](x => {
      Vectors.dense(2 * ThreadLocalRandom.current().nextDouble() - 1.0, x.toArray :_*)
    })

    val withIrrelevant = multiClassData.withColumn(
      "features",
      addRandom(multiClassData("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val trainedForMatrix: Map[String, Vector] =  new LogisticMatrixDSVRGD()
      .setRegParam(0.03).setElasticNetParam(1.0).setTol(1e-8).setLocalMinibatchSize(2)
      .fit(withIrrelevant)
      .nested.mapValues(_.getCoefficients)

    val hiddenModels = Map(
      "0" -> hiddenModel,
      "1" -> hiddenModel2,
      "2" -> hiddenModel3
    )

    trainedForMatrix.foreach(pair => {
      val v = pair._2
      Math.abs(v(0)) should be <= 1.0e-3
      Math.abs(v(1)) should be > 0.0
      Math.abs(v(2)) should be > 0.0

      cosineDistance(Vectors.dense(v.toArray.drop(1)), hiddenModels(pair._1)) should be <= 0.001
    })
  }

  "Regularization " should " eliminate correlated" in {

    val noise = 0.05

    val addCorrelated = functions.udf[Vector,Vector](x => {
      Vectors.dense(x.toArray(0) / 1.5  +  2 * noise * ThreadLocalRandom.current().nextDouble() - noise, x.toArray :_*)
    })

    val withCorrelated = multiClassData.withColumn(
      "features",
      addCorrelated(multiClassData("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val trainedForMatrix: Map[String, Vector] = new LogisticMatrixDSVRGD()
        .setLocalMinibatchSize(1)
      .setRegParam(0.03).setElasticNetParam(1.0).setTol(1e-8).setMaxIter(200).setLocalMinibatchSize(2)
      .fit(withCorrelated)
      .nested.mapValues(_.getCoefficients)

    val hiddenModels = Map(
      "0" -> hiddenModel,
      "1" -> hiddenModel2,
      "2" -> hiddenModel3
    )

    trainedForMatrix.foreach(pair => {
      val v = pair._2
      Math.abs(v(0)) should be <= 1.0e-3
      Math.abs(v(1)) should be > 0.0
      Math.abs(v(2)) should be > 0.0

      cosineDistance(Vectors.dense(v.toArray.drop(1)), hiddenModels(pair._1)) should be <= 0.001
    })
  }

  "Regularization " should " remove intercept by default" in {

    val addIntercept = functions.udf[Vector,Vector](Interceptor.appendBias)

    val withIntercept = multiClassData.withColumn(
      "features",
      addIntercept(multiClassData("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val trainedForMatrix: Map[String, Vector] = new LogisticMatrixDSVRGD()
      .setRegParam(0.035).setElasticNetParam(1.0).setTol(1e-8).setLocalMinibatchSize(2)
      .fit(withIntercept)
      .nested.mapValues(_.getCoefficients)

    val hiddenModels = Map(
      "0" -> hiddenModel,
      "1" -> hiddenModel2,
      "2" -> hiddenModel3
    )

    trainedForMatrix.foreach(pair => {
      val v = pair._2
      Math.abs(v(0)) should be > 0.0
      Math.abs(v(1)) should be > 0.0
      Math.abs(v(2)) should be <= 0.0

      cosineDistance(Vectors.dense(v.toArray.dropRight(1)), hiddenModels(pair._1)) should be <= 0.001
    })
  }


  "Regularization " should " keep intercept if asked" in {

    val addIntercept = functions.udf[Vector,Vector](Interceptor.appendBias)

    val withIntercept = multiClassData.withColumn(
      "features",
      addIntercept(multiClassData("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

    val trainedForMatrix: Map[String, Vector] = new LogisticMatrixDSVRGD()
      .setRegParam(0.03).setElasticNetParam(1.0).setTol(1e-7).setLastIsIntercept(true).setLocalMinibatchSize(2)
      .fit(withIntercept)
      .nested.mapValues(_.getCoefficients)

    val hiddenModels = Map(
      "0" -> hiddenModel,
      "1" -> hiddenModel2,
      "2" -> hiddenModel3
    )

    trainedForMatrix.foreach(pair => {
      val v = pair._2
      Math.abs(v(0)) should be > 0.0
      Math.abs(v(1)) should be > 0.0
      Math.abs(v(2)) should be > 0.0

      cosineDistance(Vectors.dense(v.toArray.dropRight(1)), hiddenModels(pair._1)) should be <= 0.001
    })
  }

  "DSVRGD " should " add loss history to summary" in {
    interceptedSgdModel.summary(DSVRGD.LossHistory).count() should be > 0L
  }

  "DSVRGD " should " add label, iter and loss columns" in {
    val model = new LogisticMatrixDSVRGD().setLocalMinibatchSize(2).fit(multiClassData)
    val schema: StructType = model.summary(DSVRGD.LossHistory).schema

    schema.size should be(3L)
    schema(0).name should be("label")
    schema(1).name should be("iteration")
    schema(2).name should be("loss")
  }

  "DSVRGD " should " ommit label column for scalars" in {
    val schema: StructType = interceptedSgdModel.summary(DSVRGD.LossHistory).schema

    schema.size should be(2L)
    schema(0).name should be("iteration")
    schema(1).name should be("loss")
  }
}

object LinearDSVRGDModelsSpec extends WithTestData {
  lazy val _interceptedSgdModel = {
    val model: LinearRegressionModel = Interceptor.intercept(new LinearDSVRGD().setLastIsIntercept(true).setLocalMinibatchSize(2)).fit(interceptData)
    model
  }
  lazy val _noInterceptLogisticModel = {
    val model: LogisticRegressionModel = new LogisticDSVRGD().setLocalMinibatchSize(2).fit(noInterceptDataLogistic)
    model
  }
  lazy val _interceptLogisticModel = {
    val model = Interceptor.intercept(new LogisticDSVRGD().setLastIsIntercept(true).setLocalMinibatchSize(2)).fit(interceptDataLogistig)
    model
  }
}
