package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains classes supporting model evaluation.
  */

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.odkl.CrossValidator.FoldsAssigner
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Transformer}
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, functions}

/**
  * Base class for evaluators. It is expected that evaluators group data into
  * some groups and then evaluate metrics for each of the groups.
  */
abstract class Evaluator[S <: Evaluator[S]](override val uid: String)
  extends Transformer with HasLabelCol with HasPredictionCol{

  def this() = this(Identifiable.randomUID("evaluator"))



  def copy(extra: ParamMap): S


  def setPredictionColumn(value: String) : this.type = set(predictionCol, value)

  def setLabelColumn(value: String) : this.type = set(labelCol, value)
}

object Evaluator extends Serializable {
  /**
    * Fit and then evaluate model. Results of evaluation is stored into a dedicated summary block.
    *
    * @param estimator Used to fit the model
    * @param evaluator Used to evaluate the model.
    * @return Estimator which returns a model fit by nested predictor with extra summary block for metrics, produced by
    *         evaluator.
    */
  def evaluate[M <: ModelWithSummary[M], E <: Evaluator[E]](estimator: SummarizableEstimator[M], evaluator: E):
  SummarizableEstimator[M]
  = {
    UnwrappedStage.modelOnly(estimator, new EvaluatingTransformer[M, E](evaluator))
  }

  /**
    * Performs a cross validation given predictor and evaluator. Returns a model with summary blocks extended
    * with foldNum column.
    *
    * Split into folds is done based on the hash of entire row.
    *
    * @param estimator  Nested predictor for fitting the model.
    * @param evaluator  Evaluator for creating a metric.
    * @param numFolds   Number of folds for validation (defeult 10)
    * @param parallel   Whenever to train and evaluate folds in parallel.
    * @param cacheForks Whenever to cache forks before iterating
    * @return Estimator which returns a model fit by the nested predictor on the entire dataset with summary blocks
    *         extended with numFolds column.
    */
  def crossValidate[M <: ModelWithSummary[M], E <: Evaluator[E]]
  (
    estimator: SummarizableEstimator[M],
    evaluator: E,
    numFolds: Int = 10,
    parallel: Boolean = false,
    cacheForks: Boolean = false):
  SummarizableEstimator[M]
  = {
    addFolds(
      validateInFolds(estimator, evaluator, numFolds, parallel, cacheForks),
      new FoldsAssigner().setNumFolds(numFolds))
  }

  /**
    * Performs a cross validation given predictor and evaluator. Returns a model with summary blocks extended
    * with foldNum column.
    *
    * Split into folds is expected to be done externaly.
    *
    * @param estimator  Nested predictor for fitting the model.
    * @param evaluator  Evaluator for creating a metric.
    * @param numFolds   Number of folds for validation (defeult 10)
    * @param parallel   Whenever to train and evaluate folds in parallel.
    * @param cacheForks Whenever to cache forks before iterating
    * @return Estimator which returns a model fit by the nested predictor on the entire dataset with summary blocks
    *         extended with numFolds column.
    */
  def validateInFolds[M <: ModelWithSummary[M], E <: Evaluator[E]]
  (
    estimator: SummarizableEstimator[M],
    evaluator: E,
    numFolds: Int = 10,
    parallel: Boolean = false,
    cacheForks: Boolean = false):
  CrossValidator[M]
  = {
    val estimatorOnTrain = UnwrappedStage.dataOnly(estimator, new TrainOnlyFilter())

    val folder: FoldsAssigner = new FoldsAssigner()
    val validator: CrossValidator[M] = new CrossValidator[M](evaluate(
      estimatorOnTrain, evaluator))

    validator
      .set(validator.numFolds, numFolds)
      .set(validator.trainParallel, parallel)
      .setCacheForks(cacheForks)
  }

  /**
    * Adds folds (foldNum column) to the dataset before passing it to the nested estimator.
    *
    * @param estimator Nested predictor for  fitting the model
    * @param folder    Transformer adding folds (by default based on row hash)
    * @return Estimator returning a model fit by nested predictor on a dataset with extra foldNum column
    */
  def addFolds[M <: ModelWithSummary[M]]
  (
    estimator: SummarizableEstimator[M],
    folder: FoldsAssigner = new FoldsAssigner()) = {
    UnwrappedStage.dataOnly(estimator, folder)
  }

  /**
    * Utility used for transparent injection of the evaluator into training chain. Evaluator is applied only while
    * fitting (it adds an extra summary block), but has no other traces in the final model (does not affect predictions).
    *
    * @param evaluator Evaluator for calculating metrics)
    */
  class EvaluatingTransformer[M <: ModelWithSummary[M], E <: Evaluator[E]]
  (
    val evaluator: E,
    override val uid: String
  ) extends UnwrappedStage.ModelOnlyTransformer[M, EvaluatingTransformer[M, E]](uid) with HasMetricsBlock {

    def this(evaluator: E) =
      this(evaluator, Identifiable.randomUID("evaluatingEstimator"))

    override def copy(extra: ParamMap): EvaluatingTransformer[M, E] = {
      copyValues(new EvaluatingTransformer[M, E](evaluator.copy(extra)))
    }

    override def transformModel(model: M, originalData: DataFrame): M = {
      val predictions = model.transform(originalData)

      log.info(s"Evaluating predictions for model $model")
      model.copy(Map(metrics -> evaluator.transform(predictions)))
    }
  }

  /**
    * This is a simple workaround to add kind of grouping by test/train column for evaluators without embedded
    * support for grouping (eg. BinaryClassificationEvaluator).
    *
    * @param nested Nested evaluator to wrap around.
    */
  class TrainTestEvaluator[N <: Evaluator[N]](val nested: N) extends Evaluator[TrainTestEvaluator[N]] with HasIsTestCol {

    override def copy(extra: ParamMap): TrainTestEvaluator[N] = {
      copyValues(new TrainTestEvaluator(nested.copy(extra)), extra)
    }

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = {

      nested.transformSchema(schema)
        .add($(isTestColumn), BooleanType, nullable = false)
    }

    override def transform(dataset: DataFrame): DataFrame = {
      val test = nested.transform(dataset.filter(dataset($(isTestColumn)) === true)).withColumn($(isTestColumn), functions.lit(true))
      val train = nested.transform(dataset.filter(dataset($(isTestColumn)) === false)).withColumn($(isTestColumn), functions.lit(false))

      test.unionAll(train)
    }
  }

  /**
    * Utility used to filter out test data before passing to estimator.
    */
  class TrainOnlyFilter(override val uid: String) extends Transformer with HasIsTestCol {
    def this() = this(Identifiable.randomUID("trainOnlyFilter"))

    override def transform(dataset: DataFrame): DataFrame = {
      val col = dataset($(isTestColumn))

      dataset.filter(dataset($(isTestColumn)) === false)
    }

    override def copy(extra: ParamMap): Transformer = copyValues(new TrainOnlyFilter(), extra)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema
  }

  class PostProcessingEvaluator[E <: Evaluator[E]](nested : E, postprocessing: Estimator[_ <: Transformer])
    extends Evaluator[PostProcessingEvaluator[E]] {
    override def copy(extra: ParamMap): PostProcessingEvaluator[E] = copyValues(
      new PostProcessingEvaluator[E](nested.copy(extra), postprocessing.copy(extra)))

    override def transform(dataset: DataFrame): DataFrame = {
      val evaluated: DataFrame = nested.transform(dataset)
      postprocessing.fit(evaluated).transform(evaluated)
    }

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = postprocessing.transformSchema(nested.transformSchema(schema))
  }

  /**
    * Used in case when folding is needed, but not the evaluation
    */
  class EmptyEvaluator extends Evaluator[EmptyEvaluator] {

    override def transform(dataset: DataFrame): DataFrame =
      // Can not use empty data frame as it causes troubles on the later stages of processing.
      SparkSqlUtils.reflectionLock.synchronized(
        dataset.sqlContext.createDataFrame(
          dataset.sqlContext.sparkContext.parallelize(Seq(Row.fromSeq(Seq(1))), 1),
          new StructType().add("meaningless", IntegerType)
        )
      )

    override def transformSchema(schema: StructType): StructType = StructType(StructField("meaningless", IntegerType) :: Nil)

    override def copy(extra: ParamMap): EmptyEvaluator = defaultCopy(extra)
  }

}

