package org.apache.spark.ml.odkl

import ml.dmlc.xgboost4j.scala.spark.{TrackerConf, XGBoostEstimator => DMLCEstimator, XGBoostModel => DMLCModel}
import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.odkl.ModelWithSummary.{WithSummaryReader, WithSummaryWriter}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Light weight wrapper for DMLC xgboost4j-spark. Optimizes defaults and provides rich summary
  * extraction.
  */
class XGBoostEstimator private (private var _dlmc: DMLCEstimator, override val uid: String)
  extends SummarizableEstimator[XGBoostModel] with HasLossHistory {
  def this(dlmc: DMLCEstimator) = {
    this(
      // Use scala implementation for Rabbit by default. Python sucks.
      dlmc.set(dlmc.trackerConf, if (dlmc.isSet(dlmc.trackerConf)) dlmc.get(dlmc.trackerConf).get else new TrackerConf(30000, "scala")),
      Identifiable.randomUID("xgboostEstimatorWrapper"))
  }

  /**
    * For serialization only
    */
  @Deprecated def this(uid: String) = this(null, uid)

  def this(params: (String, Any)*) = this(new DMLCEstimator(params.toMap))

  def dlmc: DMLCEstimator = this._dlmc

  def setUseExternalMemory(value: Boolean): this.type = {
    dlmc.set(dlmc.useExternalMemory, value)
    this
  }

  def setTrackerConf(workerConnectionTimeout: Long, trackerImpl: String): this.type = {
    dlmc.set(dlmc.trackerConf, new TrackerConf(workerConnectionTimeout, trackerImpl))
    this
  }

  def setTrainTestRation(value: Double): this.type = {
    dlmc.set(dlmc.trainTestRatio, value)
    this
  }

  def setNumClasses(value: Int): this.type = {
    dlmc.set(dlmc.numClasses, value)
    this
  }

  def setObjective(value: String): this.type = {
    dlmc.set(dlmc.objective, value)
    this
  }

  def setBaseScore(value: Double): this.type = {
    dlmc.set(dlmc.baseScore, value)
    this
  }

  def setEvalMetric(value: String): this.type = {
    dlmc.set(dlmc.evalMetric, value)
    this
  }

  def setGroupData(value: Seq[Int]*): this.type = {
    dlmc.set(dlmc.groupData, value)
    this
  }

  def setBaseMarginCol(value: String): this.type = {
    dlmc.set(dlmc.baseMarginCol, value)
    this
  }

  def setWeightCol(value: String): this.type = {
    dlmc.set(dlmc.weightCol, value)
    this
  }

  def setNumEarlyStoppingRounds(value: Int): this.type = {
    dlmc.set(dlmc.numEarlyStoppingRounds, value)
    this
  }


  def setNumRounds(value: Int): this.type = {
    dlmc.set(dlmc.round, value)
    this
  }

  def setNumWorkers(value: Int): this.type = {
    dlmc.set(dlmc.nWorkers, value)
    this
  }

  def setNumThreadsPerTask(value: Int): this.type = {
    dlmc.set(dlmc.numThreadPerTask, value)
    this
  }

  def setSilent(value: Boolean): this.type = {
    dlmc.set(dlmc.silent, if (value) 1 else 0)
    this
  }

  def setCustomObjective(value: ObjectiveTrait): this.type = {
    dlmc.set(dlmc.customObj, value)
    this
  }

  def setCustomEvaluation(value: EvalTrait): this.type = {
    dlmc.set(dlmc.customEval, value)
    this
  }

  def setMissing(value: Float): this.type = {
    dlmc.set(dlmc.missing, value)
    this
  }

  def setTimeoutRequestWorkers(value: Long): this.type = {
    dlmc.set(dlmc.timeoutRequestWorkers, value)
    this
  }

  def setCheckpointPath(value: String): this.type = {
    dlmc.set(dlmc.checkpointPath, value)
    this
  }

  def setCheckpointInterval(value: Int): this.type = {
    dlmc.set(dlmc.checkpointInterval, value)
    this
  }

  def setSeed(value: Long): this.type = {
    dlmc.set(dlmc.seed, value)
    this
  }

  def setBoosterType(value: String): this.type = {
    dlmc.set(dlmc.boosterType, value)
    this
  }

  def setEta(value: Double): this.type = {
    dlmc.set(dlmc.eta, value)
    this
  }

  def setGamma(value: Double): this.type = {
    dlmc.set(dlmc.gamma, value)
    this
  }

  def setMaxDepth(value: Int): this.type = {
    dlmc.set(dlmc.maxDepth, value)
    this
  }

  def setMinChildWeight(value: Double): this.type = {
    dlmc.set(dlmc.minChildWeight, value)
    this
  }

  def setMaxDeltaStep(value: Double): this.type = {
    dlmc.set(dlmc.maxDeltaStep, value)
    this
  }

  def setSubSample(value: Double): this.type = {
    dlmc.set(dlmc.subSample, value)
    this
  }

  def setColSampleByTree(value: Double): this.type = {
    dlmc.set(dlmc.colSampleByTree, value)
    this
  }

  def setColSampleByLevel(value: Double): this.type = {
    dlmc.set(dlmc.colSampleByLevel, value)
    this
  }

  def setLambda(value: Double): this.type = {
    dlmc.set(dlmc.lambda, value)
    this
  }

  def setAlpha(value: Double): this.type = {
    dlmc.set(dlmc.alpha, value)
    this
  }

  def setTreeMethod(value: String): this.type = {
    dlmc.set(dlmc.treeMethod, value)
    this
  }

  def setGrowthPolicy(value: String) : this.type = {
    dlmc.set(dlmc.growthPolicty, value)
    this
  }

  def setMaxBeens(value: Int) : this.type = {
    dlmc.set(dlmc.maxBins, value)
    this
  }

  def setSketchEps(value: Double) : this.type = {
    dlmc.set(dlmc.sketchEps, value)
    this
  }

  def setScalePosWeight(value: Double) : this.type = {
    dlmc.set(dlmc.scalePosWeight, value)
    this
  }

  def setSampleType(value: String) : this.type = {
    dlmc.set(dlmc.sampleType, value)
    this
  }

  def setNormalizeType(value: String) : this.type = {
    dlmc.set(dlmc.normalizeType, value)
    this
  }

  def setRateDrop(value: Double) : this.type = {
    dlmc.set(dlmc.rateDrop, value)
    this
  }

  def setSkipDrop(value: Double) : this.type = {
    dlmc.set(dlmc.skipDrop, value)
    this
  }

  def setLambdaBias(value: Double) : this.type = {
    dlmc.set(dlmc.lambdaBias, value)
    this
  }

  override def copy(extra: ParamMap): SummarizableEstimator[XGBoostModel] = new XGBoostEstimator(
    dlmc.copy(extra))

  override def fit(dataset: Dataset[_]): XGBoostModel = {
    val model = try {
      new XGBoostModel(dlmc.fit(dataset))
    } catch {
      case ex: Exception =>
        logError("First boosting attempt failed, retrying. " + ex.getMessage)
        new XGBoostModel(dlmc.fit(dataset))
    }

    val sqlc = dataset.sqlContext
    val sc = sqlc.sparkContext

    import sqlc.implicits._

    val lossHistoryData: DataFrame = model.dlmc.summary.testObjectiveHistory.map(test =>
      sc.parallelize(
        test.zip(model.dlmc.summary.trainObjectiveHistory).zipWithIndex.map(x => (x._2, x._1._2, x._1._1)), 1)
        .toDF(iteration, loss, testLoss))
      .getOrElse(sc.parallelize(model.dlmc.summary.trainObjectiveHistory.zipWithIndex.map(x => x._2 -> x._1), 1)
        .toDF(iteration, loss))

    model.copy(Map(lossHistory -> lossHistoryData)).setParent(this)
  }

  override def transformSchema(schema: StructType): StructType = dlmc.transformSchema(schema)
}

class XGBoostModel(private var _dlmc: DMLCModel, override val uid: String) extends ModelWithSummary[XGBoostModel]
  with PredictorParams {

  def dlmc: DMLCModel = this._dlmc

  /**
    * For serialization only
    */
  @Deprecated def this(uid: String) = this(null, uid)

  def this(dlmc: DMLCModel) = this(dlmc, Identifiable.randomUID("xgboostModelWrapper"))

  override protected def create(): XGBoostModel = new XGBoostModel(dlmc.copy(ParamMap()))

  override def transform(dataset: Dataset[_]): DataFrame = dlmc.transform(dataset)

  override def transformSchema(schema: StructType): StructType = dlmc.transformSchema(schema)

  override def write: WithSummaryWriter[XGBoostModel] = new ModelWithSummary.WithSummaryWriter[XGBoostModel](this) {
    protected override def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      dlmc.write.save(s"$path/dlmc")
    }
  }
}

object XGBoostModel extends MLReadable[XGBoostModel] {
  override def read: MLReader[XGBoostModel] = new WithSummaryReader[XGBoostModel]() {
    override def load(path: String): XGBoostModel = {
      super.load(path) match {
        case original: XGBoostModel =>
          original._dlmc = DefaultParamsReader.loadParamsInstance(s"$path/dlmc", sc).asInstanceOf[DMLCModel]
          original
      }
    }
  }
}