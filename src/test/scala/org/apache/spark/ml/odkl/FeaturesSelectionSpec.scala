package org.apache.spark.ml.odkl

import java.util.concurrent.ThreadLocalRandom

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.odkl.Evaluator.EmptyEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions
import org.scalatest.FlatSpec


/**
  * Created by vyacheslav.baranov on 02/12/15.
  */
class FeaturesSelectionSpec extends FlatSpec with TestEnv with WithTestData with org.scalatest.Matchers with SQLOperations with HasFeaturesSignificance {

  lazy val withIrrelevant = FeaturesSelectionSpec._withIrrelevant
  lazy val withCorellated = FeaturesSelectionSpec._withCorellated
  lazy val withBoth = FeaturesSelectionSpec._withBoth

  "Selector " should " should eliminate irrelevant" in {

    val estimator = new LogisticRegressionLBFSG().setRegParam(0.01)

    val model = SignificantFeatureSelector.select(
      estimator = estimator,
      selector = Evaluator.crossValidate(
        estimator = estimator.copy(ParamMap(estimator.elasticNetParam -> 1.0)),
        evaluator = new EmptyEvaluator(),
        numFolds = 5
      ),
      minSignificance = 15)
      .fit(withIrrelevant)

    model.getCoefficients(0) should be (0)
    Math.abs(model.getCoefficients(1)) should be > 0.0
    Math.abs(model.getCoefficients(2)) should be > 0.0

    val rawModel = estimator.fit(withIrrelevant)

    Math.abs(rawModel.getCoefficients(0)) should be > 0.0
  }

  "Selector " should " should eliminate correlated" in {

    val estimator = new LogisticRegressionLBFSG().setRegParam(0.01)

    val model = SignificantFeatureSelector.select(
      estimator = estimator,
      selector = Evaluator.crossValidate(
        estimator = estimator.copy(ParamMap(estimator.elasticNetParam -> 1.0)),
        evaluator = new EmptyEvaluator(),
        numFolds = 5
      ),
      minSignificance = 10)
      .fit(withCorellated)

    model.summary(featuresSignificance).show(10)

    model.getCoefficients(0) should be (0)
    Math.abs(model.getCoefficients(1)) should be > 0.0
    Math.abs(model.getCoefficients(2)) should be > 0.0

    val rawModel = estimator.fit(withCorellated)

    Math.abs(rawModel.getCoefficients(0)) should be > 0.0
  }

  lazy val linearModel: LogisticRegressionModel = SignificantFeatureSelector.select(
    estimator = new LogisticRegressionLBFSG().setRegParam(0.01),
    selector = Evaluator.crossValidate(
      estimator = new LogisticRegressionLBFSG().setRegParam(0.01).setElasticNetParam(1.0),
      evaluator = new EmptyEvaluator(),
      numFolds = 5
    ),
    minSignificance = 10)
    .fit(withBoth)

  "Selector " should " should eliminate both" in {

    linearModel.summary(featuresSignificance).show(10)

    linearModel.getCoefficients(0) should be (0)
    linearModel.getCoefficients(1) should be (0)
    Math.abs(linearModel.getCoefficients(2)) should be > 0.0
    Math.abs(linearModel.getCoefficients(3)) should be > 0.0
  }

  "Selector " should " should add summary block" in {

    val summary = linearModel.summary(featuresSignificance)

    summary.count should be(4)

    summary.where("index = 0").select(significance).rdd.map(_.getDouble(0)).first() should be < 14.0
    summary.where("index = 1").select(significance).rdd.map(_.getDouble(0)).first() should be < 14.0
    summary.where("index = 2").select(significance).rdd.map(_.getDouble(0)).first() should be > 14.0
    summary.where("index = 3").select(significance).rdd.map(_.getDouble(0)).first() should be > 14.0


    val foundModel = Vectors.dense(
      summary.where("index = 2").select(average).rdd.map(_.getDouble(0)).first(),
      summary.where("index = 3").select(average).rdd.map(_.getDouble(0)).first()
    )
    val deviation: Double = cosineDistance(hiddenModel, foundModel)

    Math.abs(deviation) should be <= 0.01

    summary.schema.fieldNames should be(Array(feature_index, feature_name, average, stdDev, count, significance))
  }

  "Model " should " predict classes" in {
    val model = linearModel

    val auc = new BinaryClassificationMetrics(
      model.transform(withBoth)
        .select(linearModel.getPredictionCol, linearModel.getLabelCol).rdd
        .map(r => (r.getDouble(0), r.getDouble(1)))).areaUnderROC()


    auc should be >= 0.99
  }
}

object FeaturesSelectionSpec extends WithTestData {
  val addRandom = functions.udf[Vector,Vector](x => {
    Vectors.dense(2 * ThreadLocalRandom.current().nextDouble() - 1.0, x.toArray :_*)
  })

  val addCorellated = functions.udf[Vector,Vector](x => {
    val sign = if(ThreadLocalRandom.current().nextBoolean()) 1 else -1
    Vectors.dense(x(x.size - 1) + sign * ThreadLocalRandom.current().nextDouble(0.2,0.5), x.toArray :_*)
  })

  lazy val _withIrrelevant = noInterceptDataLogistic.withColumn(
    "features",
    addRandom(noInterceptDataLogistic("features")).as("features", new AttributeGroup("features", 3).toMetadata()))

  lazy val _withCorellated = noInterceptDataLogistic.withColumn(
    "features",
    addCorellated(noInterceptDataLogistic("features")).as("features", new AttributeGroup("features", 3).toMetadata())
  )

  lazy val _withBoth = _withIrrelevant.withColumn(
    "features",
    addCorellated(_withIrrelevant("features")).as("features", new AttributeGroup("features", 4).toMetadata())
  )
}


