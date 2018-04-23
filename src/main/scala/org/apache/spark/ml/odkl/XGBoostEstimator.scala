package org.apache.spark.ml.odkl

import java.io.{File, FileWriter}

import ml.dmlc.xgboost4j.scala.spark.params.{BoosterParams, GeneralParams, LearningTaskParams}
import ml.dmlc.xgboost4j.scala.spark.{TrackerConf, XGBoostEstimator => DMLCEstimator, XGBoostModel => DMLCModel}
import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.attribute.{AttributeGroup, BinaryAttribute, NominalAttribute}
import org.apache.spark.ml.odkl.ModelWithSummary.{Block, WithSummaryReader, WithSummaryWriter}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Light weight wrapper for DMLC xgboost4j-spark. Optimizes defaults and provides rich summary
  * extraction.
  */
class XGBoostEstimator(override val uid: String)
  extends SummarizableEstimator[XGBoostModel]
    with LearningTaskParams with GeneralParams with BoosterParams with PredictorParams
    with HasLossHistory with HasFeaturesSignificance with DefaultParamsWritable {

  val addRawTrees = new BooleanParam(this, "addRawTrees",
    "Whenever to add raw trees block to model summary.")

  val addSignificance = new BooleanParam(this, "addSignificance",
    "Whenever to add feature significance block to model summary.")

  def setAddSignificance(value: Boolean): this.type = set(addSignificance, value)

  def setAddRawTrees(value: Boolean): this.type = set(addRawTrees, value)

  setDefault(addRawTrees -> true, addSignificance -> true,
    trackerConf -> new TrackerConf(30000, "scala"))

  def this() =
    this(
      // Use scala implementation for Rabbit by default. Python sucks.
      //set(trackerConf, if (dlmc.isSet(dlmc.trackerConf)) dlmc.get(dlmc.trackerConf).get else new TrackerConf(30000, "scala")) ,
      Identifiable.randomUID("xgboostEstimatorWrapper"))

  def setFeatureCol(value: String): this.type = set(featuresCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setUseExternalMemory(value: Boolean): this.type = set(useExternalMemory, value)

  def setTrackerConf(workerConnectionTimeout: Long, trackerImpl: String): this.type = set(trackerConf, new TrackerConf(workerConnectionTimeout, trackerImpl))

  def setTrainTestRation(value: Double): this.type = set(trainTestRatio, value)

  def setNumClasses(value: Int): this.type = set(numClasses, value)

  def setObjective(value: String): this.type = set(objective, value)

  def setBaseScore(value: Double): this.type = set(baseScore, value)

  def setEvalMetric(value: String): this.type = set(evalMetric, value)

  def setGroupData(value: Seq[Int]*): this.type = set(groupData, value)

  def setBaseMarginCol(value: String): this.type = set(baseMarginCol, value)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setNumEarlyStoppingRounds(value: Int): this.type = set(numEarlyStoppingRounds, value)


  def setNumRounds(value: Int): this.type = set(round, value)

  def setNumWorkers(value: Int): this.type = set(nWorkers, value)

  def setNumThreadsPerTask(value: Int): this.type = set(numThreadPerTask, value)

  def setSilent(value: Boolean): this.type = set(silent, if (value) 1 else 0)

  def setCustomObjective(value: ObjectiveTrait): this.type = set(customObj, value)

  def setCustomEvaluation(value: EvalTrait): this.type = set(customEval, value)

  def setMissing(value: Float): this.type = set(missing, value)

  def setTimeoutRequestWorkers(value: Long): this.type = set(timeoutRequestWorkers, value)

  def setCheckpointPath(value: String): this.type = set(checkpointPath, value)

  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def setBoosterType(value: String): this.type = set(boosterType, value)

  def setEta(value: Double): this.type = set(eta, value)

  def setGamma(value: Double): this.type = set(gamma, value)

  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  def setMinChildWeight(value: Double): this.type = set(minChildWeight, value)

  def setMaxDeltaStep(value: Double): this.type = set(maxDeltaStep, value)

  def setSubSample(value: Double): this.type = set(subSample, value)

  def setColSampleByTree(value: Double): this.type = set(colSampleByTree, value)

  def setColSampleByLevel(value: Double): this.type = set(colSampleByLevel, value)

  def setLambda(value: Double): this.type = set(lambda, value)

  def setAlpha(value: Double): this.type = set(alpha, value)

  def setTreeMethod(value: String): this.type = set(treeMethod, value)

  def setGrowthPolicy(value: String): this.type = set(growthPolicty, value)

  def setMaxBeens(value: Int): this.type = set(maxBins, value)

  def setSketchEps(value: Double): this.type = set(sketchEps, value)

  def setScalePosWeight(value: Double): this.type = set(scalePosWeight, value)

  def setSampleType(value: String): this.type = set(sampleType, value)

  def setNormalizeType(value: String): this.type = set(normalizeType, value)

  def setRateDrop(value: Double): this.type = set(rateDrop, value)

  def setSkipDrop(value: Double): this.type = set(skipDrop, value)

  def setLambdaBias(value: Double): this.type = set(lambdaBias, value)

  override def copy(extra: ParamMap): SummarizableEstimator[XGBoostModel] = defaultCopy(extra)

  private def trainInternal(dataset: Dataset[_]): DMLCModel = {
    val estimator = copyValues(new DMLCEstimator(Map[String, Any]()))

    // This trick is used to support estimator re-read and reset empty group to null
    if (isSet(groupData) && $(groupData) != null && $(groupData).isEmpty) {
      estimator.set(estimator.groupData, null)
    }

    estimator.fit(dataset)
  }

  override def fit(dataset: Dataset[_]): XGBoostModel = {
    val model = try {
      new XGBoostModel(trainInternal(dataset))
    } catch {
      case ex: Exception =>
        // Yes, it might happen so that fist training attempt fail due to racing condition
        logError("First boosting attempt failed, retrying. " + ex.getMessage)
        new XGBoostModel(trainInternal(dataset))
    }

    // OK, we got the model, enrich the summary
    val sqlc = dataset.sqlContext
    val sc = sqlc.sparkContext

    import sqlc.implicits._

    var blocks = Map[Block, DataFrame]()

    // Loss history for the training process
    blocks += lossHistory -> model.dlmc.summary.testObjectiveHistory.map(test =>
      sc.parallelize(
        test.zip(model.dlmc.summary.trainObjectiveHistory).zipWithIndex.map(x => (x._2, x._1._2, x._1._1)), 1)
        .toDF(iteration, loss, testLoss))
      .getOrElse(sc.parallelize(model.dlmc.summary.trainObjectiveHistory.zipWithIndex.map(x => x._2 -> x._1), 1)
        .toDF(iteration, loss))

    if ($(addSignificance) || $(addRawTrees)) {
      // Both tree dump and features significance needs this map to produce more readable result
      val featureMap: Option[Array[(Int, String, String)]] = AttributeGroup.fromStructField(dataset.schema(getFeaturesCol)).attributes
        .map(x => x.map(attr => {
          val index = attr.index.get
          val flag: String = if (attr.isNominal) {
            attr match {
              case _: BinaryAttribute => "i"
              case nom: NominalAttribute if nom.getNumValues.exists(x => x <= 1) => "i"
              case ord: NominalAttribute if ord.isOrdinal.exists(x => x) => "int"
              case _ => "g"
            }
          } else "q"
          (index, attr.name.getOrElse(s"f$index"), flag)
        }))


      // This map can only be passed in a file
      val fileName = featureMap.map(x => {
        val fmap = new File(FileUtils.getTempDirectory, uid + "_fmap")

        val writer = new FileWriter(fmap)

        try {
          writer.append(x.map(_.productIterator.mkString("\t")).mkString("\n"))
        } finally {
          writer.close()
        }

        fmap
      })

      try {
        val fmap = fileName.map(_.getAbsolutePath).orNull

        // Features significance block
        if ($(addSignificance)) {
          blocks += featuresSignificance -> {
            val dlmcSign = model.dlmc.booster.getFeatureScore(fmap)

            val sig = featureMap
              .map(_.map(x => (x._1, x._2, dlmcSign.get(x._2).map(_.doubleValue()).getOrElse(Double.NaN))))
              .getOrElse(dlmcSign.toArray.sortBy(_._1).zipWithIndex.map(x => (x._2, x._1._1, x._2.doubleValue())))

            sc.parallelize(sig, 1).toDF(feature_index, feature_name, significance)
          }
        }

        // Raw trees block
        if ($(addRawTrees)) {
          blocks += Block("rawTrees") -> sc
            .parallelize(model.dlmc.booster.getModelDump(fmap).zipWithIndex.map(_.swap), 1)
            .toDF("index", "treeData")
        }
      } finally {
        fileName.foreach(_.delete())
      }
    }

    model.copy(blocks).setParent(this)
  }

  override def transformSchema(schema: StructType): StructType =
    copyValues(new DMLCEstimator(Map[String, Any]())).transformSchema(schema)
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

object XGBoostEstimator extends DefaultParamsReadable[XGBoostEstimator]

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