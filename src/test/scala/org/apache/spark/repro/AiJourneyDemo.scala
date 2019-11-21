package org.apache.spark.repro

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.odkl.CrossValidator.FoldsAssigner
import org.apache.spark.ml.odkl.Evaluator.TrainTestEvaluator
import org.apache.spark.ml.odkl.hyperopt._
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.param.ParamMap
import org.mlflow.tracking.creds.BasicMlflowHostCreds

/**
  * This code was used to prepare demo for AI Journey 2019. The data for the demo might be found at
  * https://cloud.mail.ru/public/A9cF/bVSWJCJgt
  */
class AiJourneyDemo extends TestEnv {

  import sqlc.implicits._

  // Feature extractor used - in order to force overfitting and avoid trivial optimal parameters
  // we add ownerId_string column and one-hot it.
  val featureExtraction = Array(
    new SQLTransformer().setStatement("SELECT *, cast(metadata_ownerId % 257 AS string) AS ownerId_string, IF(array_contains(feedback, 'Liked'), 1.0, 0.0) AS label FROM __THIS__"),
    new NullToDefaultReplacer(),
    new AutoAssembler()
      .setColumnsToExclude("date", "instanceId_userId", "instanceId_objectId", "feedback", "label", "ImageId")
      .setOutputCol("features"),
    new SQLTransformer().setStatement("SELECT date, features, cast(label AS double) AS label FROM __THIS__"))

  // The mode to train
  val logReg = new LogisticRegressionLBFSG()
    .setRegularizeLast(false)

  // The data to train and test on.
  val train = sqlc.read
    .option("basePath", "sna2019/offlineTrain")
    .parquet("sna2019/offlineTrain/date={2018-02-01,2018-03-21}")

  // For speedup just use a  single fold
  val numFolds = 1

  val evaluatedModel = Evaluator.addFolds(
    estimator = Evaluator.validateInFolds(
      estimator = Scaler.scale(Interceptor.intercept(logReg)),
      evaluator = new TrainTestEvaluator(new BinaryClassificationEvaluator()),
      numFolds = numFolds,
      numThreads = numFolds)
      // This is how we can achieve time-based validation.
      .setTestSetExpression("date > '2018-03-19'")
      // Metrics to report to MLFlow
      .setExtractExpression(
      "SELECT metric, value AS value, isTest, foldNum AS step FROM __THIS__ WHERE `x-value` IS NULL")
      .setAddGlobal(false)
      .setEnableDive(false),
    folder = new FoldsAssigner().setNumFolds(numFolds))

  // Params grid for grid search
  val params = new StableOrderParamGridBuilder()
    .addGrid(logReg.regParam, Array(0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95))
    .addGrid(logReg.elasticNetParam, Array(0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95))
    .build()

  val gridSearch = new GridSearch(evaluatedModel)
    .setEstimatorParamMaps(params)

  // Random search confirguration
  val stochasticHyperopt = new StochasticHyperopt[LogisticRegressionModel](evaluatedModel)
    .setParamDomains(
      ParamDomainPair(logReg.regParam, DoubleRangeDomain(0.0, 1.0)),
      ParamDomainPair(logReg.elasticNetParam, DoubleRangeDomain(0.0, 1.0)))
    .setMaxIter(params.length - 1)

  // Stochastic search with Gaussian process
  val gaussianProcessOptimizer = stochasticHyperopt.copy(
    ParamMap(stochasticHyperopt.searchMode -> BayesianParamOptimizer.GAUSSIAN_PROCESS)
  ).setEpsilonGreedy(0.3)

  // Method for fitting a model with certain optimizer
  def fitAndSave(name: String, optimizer: HyperparametersOptimizer[LogisticRegressionModel]): Unit = {
    val pipeline = new Pipeline().setStages(featureExtraction :+
      UnwrappedStage.cacheAndMaterialize(
        optimizer
          .setMetricsExpression("SELECT AVG(value) FROM __THIS__ WHERE metric = 'auc' AND isTest")
          .setParamNames(logReg.regParam -> "regParam", logReg.elasticNetParam -> "elasticNet")
          .setNumThreads(10))
    )

    // This is the way to setup context for tracing metric
    implicit val reproContext: MlFlowReproContext = new MlFlowReproContext(creds = new BasicMlflowHostCreds(
      "https://mlflow.tools.kc.odkl.ru/", null, null, null, true),
      basePath = "mlFlowTest", experiment = "AI Journey (AUC)")(sqlc.sparkSession)


    val result = pipeline.reproducableFit(train)

    result.write.overwrite().save(s"aiJouneyAuc_run3/$name")
  }

  Seq(
    "grid" -> gridSearch,
    "stochastic" -> stochasticHyperopt,
    "gaussian" -> gaussianProcessOptimizer
  ).foreach(x => fitAndSave(x._1, x._2))

}
