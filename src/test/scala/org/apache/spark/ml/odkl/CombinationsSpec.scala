package org.apache.spark.ml.odkl

import java.io.File

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.odkl.PartitionedRankingEvaluator._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.functions
import org.scalatest.FlatSpec

/**
  * Created by dmitriybugaichenko on 26.01.16.
  */
class CombinationsSpec extends FlatSpec with TestEnv with org.scalatest.Matchers
  with SQLOperations with WithTestData with HasWeights with HasMetricsBlock {

  lazy val typeSelectingModel = CombinedModel.perType(new LogisticRegressionLBFSG()).fit(typedWithLabels)

  lazy val multiClassModel: LinearCombinationModel[LogisticRegressionModel] =
    CombinedModel.linearCombination(new LogisticRegressionLBFSG()).fit(withClass)

  lazy val matrixData = new MultinominalExtractorModel()
    .setInputCol("classes")
    .setOutputCol("label")
    .setValues("Positive", "Negative")
    .transform(withClass)

  lazy val matrixModel: LinearCombinationModel[LogisticRegressionModel] = new MatrixLBFGS()
    .fit(matrixData)

  lazy val typeWhenClass = CombinedModel.perType(
    CombinedModel.linearCombination(new LogisticRegressionLBFSG()).setTrainParallel(true)).setTrainParallel(true)
    .fit(withTypeAndClass)

  lazy val scaled = CombinedModel.perType(
    Scaler.scaleComposite[LogisticRegressionModel, LinearCombinationModel[LogisticRegressionModel]](
      estimator = CombinedModel.linearCombination(new LogisticRegressionLBFSG())))
    .fit(withTypeAndClass)

  lazy val intercepted = CombinedModel.perType(
    Interceptor.interceptComposite[LogisticRegressionModel, LinearCombinationModel[LogisticRegressionModel]](
      estimator = CombinedModel.linearCombination(new LogisticRegressionLBFSG())))
    .fit(withTypeAndClass)

  lazy val scaledAndIntercepted = CombinedModel.perType(
    Scaler.scaleComposite[LogisticRegressionModel, LinearCombinationModel[LogisticRegressionModel]](
      Interceptor.interceptComposite[LogisticRegressionModel, LinearCombinationModel[LogisticRegressionModel]](
        estimator = CombinedModel.linearCombination(new LogisticRegressionLBFSG()))))
    .fit(withTypeAndClass)

  lazy val reReadModel = {
    val directory = new File(FileUtils.getTempDirectory, typeWhenClass.uid)
    try {
      typeWhenClass.save(directory.getAbsolutePath)

      val reReadModel = CombinedModel.perTypeReader(
        CombinedModel.linearCombinationReader(
          ModelWithSummary.reader(classOf[LogisticRegressionModel]))).load(directory.getAbsolutePath)

      reReadModel.summary.blocks.foreach(x => x._2.cache().count())

      reReadModel
    } finally {
      FileUtils.deleteDirectory(directory)
    }
  }

  "Type selector " should " find types" in {
    typeSelectingModel.nested.keys.toSeq.sorted should be(Seq("Direct", "Inverse"))
  }

  "Type selector " should " train inverted models" in {
    cosineDistance(
      typeSelectingModel.nested("Direct").getCoefficients,
      typeSelectingModel.nested("Inverse").getCoefficients
    ) should be >= 0.9
  }

  "Type selector " should " be able to predict classes" in {

    val metrics = new BinaryClassificationEvaluator().transform(typeSelectingModel.transform(typedWithLabels))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }

  "Class multiplexer " should " find types" in {
    multiClassModel.nested.keys.toSeq.sorted should be(Seq("Negative", "Positive"))
  }


  "Class multiplexer " should " filter types" in {
    val model = CombinedModel.linearCombination(new LogisticRegressionLBFSG()).setClassesToIgnore("Negative").fit(withClass)

    model.nested.keys.toSeq.sorted should be(Seq("Positive"))
  }

  "Class multiplexer " should " rename types" in {
    val model = CombinedModel.linearCombination(new LogisticRegressionLBFSG()).setClassesMap("Negative" -> "Dislike").fit(withClass)

    model.nested.keys.toSeq.sorted should be(Seq("Dislike", "Positive"))
  }

  "Class multiplexer " should " train inverted models" in {
    cosineDistance(
      multiClassModel.nested("Positive").getCoefficients,
      multiClassModel.nested("Negative").getCoefficients
    ) should be >= 0.9
  }

  "Class multiplexer " should " be able to predict positive class" in {

    val model = multiClassModel.copy(ParamMap(multiClassModel.predictCombinations -> Array(Map("Positive" -> 1.0, "Negative" -> 0.0))))

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(noInterceptDataLogistic))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }

  "Class multiplexer " should " be able to predict negative class" in {

    val model = multiClassModel.copy(ParamMap(multiClassModel.predictCombinations -> Array(Map("Positive" -> 0.0, "Negative" -> -1.0))))

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(noInterceptDataLogistic))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }


  "Class multiplexer " should " be able to predict mixture" in {

    val model = multiClassModel.copy(ParamMap(multiClassModel.predictCombinations -> Array(Map("Positive" -> 1.0, "Negative" -> -1.0))))

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(noInterceptDataLogistic))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }

  "Class multiplexer " should " be able to predict vector" in {

    val model = multiClassModel

    model.setPredictVector(Map("Positive" -> 1.0), Map("Negative" -> 1.0), Map("Positive" -> 1.0, "Negative" -> -1.0)).transform(noInterceptDataLogistic)
      .select("features", "prediction")
      .collect.foreach(r => {
      val features = r.getAs[Vector](0)
      val prediction = r.getAs[Vector](1)

      val positive = model.nested("Positive").predictDirect(features)
      val negative = model.nested("Negative").predictDirect(features)

      prediction(0) should be(positive)
      prediction(1) should be(negative)
      prediction(2) should be(positive - negative)
    })
  }

  "Class multiplexer " should " add metadata when predicting vector" in {

    val model = multiClassModel

    val data = model.setPredictVector(Map("Positive" -> 1.0), Map("Negative" -> 1.0), Map("Positive" -> 1.0, "Negative" -> -1.0)).transform(noInterceptDataLogistic)

    val attirbutes = AttributeGroup.fromStructField(data.schema("prediction"))

    attirbutes.size should be(3)
    attirbutes(0).name.get should be(Map("Positive" -> 1.0).toString())
    attirbutes(1).name.get should be(Map("Negative" -> 1.0).toString())
    attirbutes(2).name.get should be(Map("Positive" -> 1.0, "Negative" -> -1.0).toString())
  }

  "MatrixLBFGS " should " be able to predict mixture" in {

    val model = matrixModel.copy(
      ParamMap(matrixModel.predictCombinations -> Array( Map("Positive" -> 1.0, "Negative" -> -1.0))))

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(noInterceptDataLogistic))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }

  "MatrixLBFGS " should " preserve summary " in {

    matrixModel.summary.blocks.size should be(1)

    val weights = matrixModel.summary.blocks(this.weights)

    val positiveWeights = weights
      .filter("classes = 'Positive'")
      .select(index, weight)
      .collect()
      .map(x => x.getInt(0) -> x.getDouble(1))
      .toMap

    positiveWeights.size should be(2)

    val coefficients: Vector = matrixModel.nested("Positive").getCoefficients

    positiveWeights(0) should be(coefficients(0))
    positiveWeights(1) should be(coefficients(1))

    val negativeWeights = weights
      .filter("classes = 'Negative'")
      .select(index, weight)
      .collect()
      .map(x => x.getInt(0) -> x.getDouble(1))
      .toMap

    positiveWeights.size should be(2)

    val coefficientsNegative: Vector = matrixModel.nested("Negative").getCoefficients

    negativeWeights(0) should be(coefficientsNegative(0))
    negativeWeights(1) should be(coefficientsNegative(1))
  }

  "MatrixLBFGS " should " preserve folds when used with cross validation " in {

    val trainer = Scaler.scaleComposite[LogisticRegressionModel, LinearCombinationModel[LogisticRegressionModel]](
      Interceptor.interceptComposite[LogisticRegressionModel, LinearCombinationModel[LogisticRegressionModel]](
        Evaluator.crossValidate(
          new MatrixLBFGS(),
          new PartitionedRankingEvaluator()
            .setGroupByColumns("isTest")
            .setMetrics(auc())
            .setModelThreshold(0.0),
          numFolds = 2
        )))

    val model = trainer.fit(matrixData.withColumn("isTest", functions.lit(0)))

    model.summary.blocks.size should be(2)

    val foldsInWeighs = model.summary.blocks(weights).select("foldNum").distinct().map(_.getInt(0)).collect().sorted
    val foldsInMetrics = model.summary.blocks(metrics).select("foldNum").distinct().map(_.getInt(0)).collect().sorted

    foldsInWeighs should be(Array(-1, 0, 1))
    foldsInMetrics should be(Array(-1, 0, 1))
  }

  "Type/class multiplexer " should " be able to predict mixture" in {

    val original = typeWhenClass
    val params = original.nested.values.map(m => m.predictCombinations -> Array(Map("Positive" -> 1.0, "Negative" -> -1.0)))
    val model = original.copy(ParamMap(params.toSeq: _*))

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(typedWithLabels))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }


  "Type/class multiplexer " should " be able to predict vector" in {

    val original = typeWhenClass
    val params = original.nested.values.map(m =>
      m.predictCombinations -> Array(Map("Positive" -> 1.0), Map("Negative" -> 1.0), Map("Positive" -> 1.0, "Negative" -> -1.0)))

    val model = original.copy(ParamMap(params.toSeq: _*))

    model.transform(typedWithLabels).select("features", "prediction", "type").collect().foreach(r => {
      val typedModel = model.nested(r.getString(2))

      val features = r.getAs[Vector](0)
      val prediction = r.getAs[Vector](1)

      val positive = typedModel.nested("Positive").predictDirect(features)
      val negative = typedModel.nested("Negative").predictDirect(features)

      prediction(0) should be(positive)
      prediction(1) should be(negative)
      prediction(2) should be(positive - negative)
    })
  }

  "Type/class multiplexer " should " add metadata when predicting vector" in {

    val original = typeWhenClass
    val params = original.nested.values.map(m =>
      m.predictCombinations -> Array(Map("Positive" -> 1.0), Map("Negative" -> 1.0), Map("Positive" -> 1.0, "Negative" -> -1.0)))

    val model = original.copy(ParamMap(params.toSeq: _*))

    val data = model.transform(typedWithLabels)

    val attirbutes = AttributeGroup.fromStructField(data.schema("prediction"))

    attirbutes.size should be(3)
    attirbutes(0).name.get should be(Map("Positive" -> 1.0).toString())
    attirbutes(1).name.get should be(Map("Negative" -> 1.0).toString())
    attirbutes(2).name.get should be(Map("Positive" -> 1.0, "Negative" -> -1.0).toString())
  }

  "Type/class multiplexer " should " be able to predict mixture after re-read" in {

    val params = reReadModel.nested.values.map(m => m.predictCombinations -> Array( Map("Positive" -> 1.0, "Negative" -> -1.0)))
    val model = reReadModel.copy(ParamMap(params.toSeq: _*))

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(typedWithLabels))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }

  "Type/class multiplexer " should " produce aggregated summary blocks" in {

    typeWhenClass.summary.blocks.size should be(1)

    val weights = typeWhenClass.summary.blocks(this.weights)

    val types = weights.select("type").distinct().map(_.getString(0)).collect().sorted
    types should be(Seq("Direct", "Inverse"))

    val classes = weights.select("classes").distinct().map(_.getString(0)).collect().sorted
    classes should be(Seq("Negative", "Positive"))

    val combinations = weights.select("type", "classes").distinct().map(x => x.getString(0) -> x.getString(1)).collect().sorted
    combinations should be(Seq(
      "Direct" -> "Negative",
      "Direct" -> "Positive",
      "Inverse" -> "Negative",
      "Inverse" -> "Positive"
    ))

    val directPositiveWeights = weights
      .filter("type = 'Direct' AND classes = 'Positive'")
      .select(index, weight)
      .collect()
      .map(x => x.getInt(0) -> x.getDouble(1))
      .toMap

    directPositiveWeights.size should be(2)

    val coefficients: Vector = typeWhenClass.nested("Direct").nested("Positive").getCoefficients

    directPositiveWeights(0) should be(coefficients(0))
    directPositiveWeights(1) should be(coefficients(1))
  }

  "Type/class multiplexer " should " preserve individual summary blocks" in {
    val directPositiveWeights = typeWhenClass.nested("Direct").nested("Positive").summary.blocks(this.weights)
      .select(index, weight)
      .collect()
      .map(x => x.getInt(0) -> x.getDouble(1))
      .toMap

    directPositiveWeights.size should be(2)

    val coefficients: Vector = typeWhenClass.nested("Direct").nested("Positive").getCoefficients

    directPositiveWeights(0) should be(coefficients(0))
    directPositiveWeights(1) should be(coefficients(1))
  }

  "Type/class multiplexer " should " produce aggregated summary blocks after re-read" in {

    reReadModel.summary.blocks.size should be(1)

    val weights = reReadModel.summary.blocks(this.weights)

    val types = weights.select("type").distinct().map(_.getString(0)).collect().sorted
    types should be(Seq("Direct", "Inverse"))

    val classes = weights.select("classes").distinct().map(_.getString(0)).collect().sorted
    classes should be(Seq("Negative", "Positive"))

    val combinations = weights.select("type", "classes").distinct().map(x => x.getString(0) -> x.getString(1)).collect().sorted
    combinations should be(Seq(
      "Direct" -> "Negative",
      "Direct" -> "Positive",
      "Inverse" -> "Negative",
      "Inverse" -> "Positive"
    ))

    val directPositiveWeights = weights
      .filter("type = 'Direct' AND classes = 'Positive'")
      .select(index, weight)
      .collect()
      .map(x => x.getInt(0) -> x.getDouble(1))
      .toMap

    directPositiveWeights.size should be(2)

    val coefficients: Vector = reReadModel.nested("Direct").nested("Positive").getCoefficients

    directPositiveWeights(0) should be(coefficients(0))
    directPositiveWeights(1) should be(coefficients(1))
  }

  "Type/class multiplexer " should " preserve individual summary blocks after re-read" in {
    val directPositiveWeights = reReadModel.nested("Direct").nested("Positive").summary.blocks(this.weights)
      .select(index, weight)
      .collect()
      .map(x => x.getInt(0) -> x.getDouble(1))
      .toMap

    directPositiveWeights.size should be(2)

    val coefficients: Vector = reReadModel.nested("Direct").nested("Positive").getCoefficients

    directPositiveWeights(0) should be(coefficients(0))
    directPositiveWeights(1) should be(coefficients(1))
  }

  "Type/class multiplexer " should " be able to predict mixture with scaler" in {

    val original = scaled
    val params = original.nested.values.map(m => m.predictCombinations -> Array(Map("Positive" -> 1.0, "Negative" -> -1.0)))
    val model = original.copy(ParamMap(params.toSeq: _*))

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(typedWithLabels))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }

  "Type/class multiplexer " should " be able to predict mixture with intercept" in {

    val original = intercepted
    val params = original.nested.values.map(m => m.predictCombinations -> Array(Map("Positive" -> 1.0, "Negative" -> -1.0)))
    val model = original.copy(ParamMap(params.toSeq: _*))

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(typedWithLabels))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }

  "Type/class multiplexer " should " be able to predict mixture with scaler and intercept" in {

    val original = scaledAndIntercepted
    val params = original.nested.values.map(m => m.predictCombinations -> Array(Map("Positive" -> 1.0, "Negative" -> -1.0)))
    val model = original.copy(ParamMap(params.toSeq: _*))

    val metrics = new BinaryClassificationEvaluator().transform(model.transform(typedWithLabels))

    metrics.filter("metric = 'auc'").groupBy().avg("value").take(1).head.getDouble(0) should be >= 0.9
  }
}
