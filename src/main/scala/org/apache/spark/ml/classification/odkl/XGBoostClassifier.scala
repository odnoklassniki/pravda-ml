package org.apache.spark.ml.classification.odkl

import java.io.{File, FileWriter}

import ml.dmlc.xgboost4j.scala.spark.{OkXGBoostClassifierParams, TrackerConf, XGBoostUtils, XGBoostClassificationModel => DMLCModel, XGBoostClassifier => DMLCEstimator}
import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.attribute.{AttributeGroup, BinaryAttribute, NominalAttribute}
import org.apache.spark.ml.classification.ProbabilisticClassifierParams
import org.apache.spark.ml.odkl.ModelWithSummary.{Block, WithSummaryReader, WithSummaryWriter}
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Light weight wrapper for DMLC xgboost4j-spark. Optimizes defaults and provides rich summary
  * extraction.
  */
class XGBoostClassifier(override val uid: String)
  extends SummarizableEstimator[XGClassificationModelWrapper]
    with OkXGBoostClassifierParams with ProbabilisticClassifierParams
    with HasLossHistory with HasFeaturesSignificance with DefaultParamsWritable {

  val addRawTrees = new BooleanParam(this, "addRawTrees",
    "Whenever to add raw trees block to model summary.")

  val addSignificance = new BooleanParam(this, "addSignificance",
    "Whenever to add feature significance block to model summary.")

  def setAddSignificance(value: Boolean): this.type = set(addSignificance, value)

  def setAddRawTrees(value: Boolean): this.type = set(addRawTrees, value)

  setDefault(
    addRawTrees -> true,
    addSignificance -> true,
    missing -> 0.0f,
    trackerConf -> new TrackerConf(30000, "scala"))

  def this() =
    this(
      // Use scala implementation for Rabbit by default. Python sucks.
      //set(trackerConf, if (dlmc.isSet(dlmc.trackerConf)) dlmc.get(dlmc.trackerConf).get else new TrackerConf(30000, "scala")) ,
      Identifiable.randomUID("xgboostEstimatorWrapper"))

  // Taken from dlmc
  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setBaseMarginCol(value: String): this.type = set(baseMarginCol, value)

  def setNumClass(value: Int): this.type = set(numClass, value)

  // setters for general params
  def setNumRound(value: Int): this.type = set(numRound, value)

  def setNumWorkers(value: Int): this.type = set(numWorkers, value)

  def setNthread(value: Int): this.type = set(nthread, value)

  def setUseExternalMemory(value: Boolean): this.type = set(useExternalMemory, value)

  def setSilent(value: Int): this.type = set(silent, value)

  def setMissing(value: Float): this.type = set(missing, value)

  def setTimeoutRequestWorkers(value: Long): this.type = set(timeoutRequestWorkers, value)

  def setCheckpointPath(value: String): this.type = set(checkpointPath, value)

  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def setEta(value: Double): this.type = set(eta, value)

  def setGamma(value: Double): this.type = set(gamma, value)

  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  def setMinChildWeight(value: Double): this.type = set(minChildWeight, value)

  def setMaxDeltaStep(value: Double): this.type = set(maxDeltaStep, value)

  def setSubsample(value: Double): this.type = set(subsample, value)

  def setColsampleBytree(value: Double): this.type = set(colsampleBytree, value)

  def setColsampleBylevel(value: Double): this.type = set(colsampleBylevel, value)

  def setLambda(value: Double): this.type = set(lambda, value)

  def setAlpha(value: Double): this.type = set(alpha, value)

  def setTreeMethod(value: String): this.type = set(treeMethod, value)

  def setGrowPolicy(value: String): this.type = set(growPolicy, value)

  def setMaxBins(value: Int): this.type = set(maxBins, value)

  def setSketchEps(value: Double): this.type = set(sketchEps, value)

  def setScalePosWeight(value: Double): this.type = set(scalePosWeight, value)

  def setSampleType(value: String): this.type = set(sampleType, value)

  def setNormalizeType(value: String): this.type = set(normalizeType, value)

  def setRateDrop(value: Double): this.type = set(rateDrop, value)

  def setSkipDrop(value: Double): this.type = set(skipDrop, value)

  def setLambdaBias(value: Double): this.type = set(lambdaBias, value)

  // setters for learning params
  def setObjective(value: String): this.type = set(objective, value)

  def setBaseScore(value: Double): this.type = set(baseScore, value)

  def setEvalMetric(value: String): this.type = set(evalMetric, value)

  def setTrainTestRatio(value: Double): this.type = set(trainTestRatio, value)

  def setNumEarlyStoppingRounds(value: Int): this.type = set(numEarlyStoppingRounds, value)

  def setCustomObj(value: ObjectiveTrait): this.type = set(customObj, value)

  def setCustomEval(value: EvalTrait): this.type = set(customEval, value)

  // Added by OK

  def setFeatureCol(value: String): this.type = set(featuresCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setTrackerConf(workerConnectionTimeout: Long, trackerImpl: String): this.type = set(trackerConf, new TrackerConf(workerConnectionTimeout, trackerImpl))

  def setTrainTestRation(value: Double): this.type = set(trainTestRatio, value)

  def setNumRounds(value: Int): this.type = set(numRound, value)

  def setSilent(value: Boolean): this.type = set(silent, if (value) 1 else 0)

  def setCustomObjective(value: ObjectiveTrait): this.type = set(customObj, value)

  def setCustomEvaluation(value: EvalTrait): this.type = set(customEval, value)

  def setMaxBeens(value: Int): this.type = set(maxBins, value)


  override def copy(extra: ParamMap): SummarizableEstimator[XGClassificationModelWrapper] = defaultCopy(extra)

  private def trainInternal(dataset: Dataset[_]): DMLCModel = {
    val estimator = copyValues(new DMLCEstimator())
    estimator.fit(dataset)
  }

  override def fit(dataset: Dataset[_]): XGClassificationModelWrapper = {
    val model = try {
      new XGClassificationModelWrapper(trainInternal(dataset))
    } catch {
      case ex: Exception =>
        // Yes, it might happen so that fist training attempt fail due to racing condition
        logError("First boosting attempt failed, retrying. " + ex.getMessage)
        new XGClassificationModelWrapper(trainInternal(dataset))
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
          (index, attr.name.getOrElse(s"f$index").replaceAll(" |\t", "_"), flag)
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
            val dlmcSign = XGBoostUtils.getBooster(model.dlmc).getFeatureScore(fmap)

            val sig = featureMap
              .map(_.map(x => (x._1, x._2, dlmcSign.get(x._2).map(_.doubleValue()).getOrElse(Double.NaN))))
              .getOrElse(dlmcSign.toArray.sortBy(_._1).zipWithIndex.map(x => (x._2, x._1._1, x._2.doubleValue())))

            sc.parallelize(sig, 1).toDF(feature_index, feature_name, significance)
          }
        }

        // Raw trees block
        if ($(addRawTrees)) {
          blocks += Block("rawTrees") -> sc
            .parallelize(XGBoostUtils.getBooster(model.dlmc).getModelDump(fmap).zipWithIndex.map(_.swap), 1)
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

object XGBoostClassifier extends DefaultParamsReadable[XGBoostClassifier]

class XGClassificationModelWrapper(private var _dlmc: DMLCModel, override val uid: String) extends ModelWithSummary[XGClassificationModelWrapper]
  with PredictorParams {

  def dlmc: DMLCModel = this._dlmc

  /**
    * For serialization only
    */
  @Deprecated def this(uid: String) = this(null, uid)

  def this(dlmc: DMLCModel) = this(dlmc, Identifiable.randomUID("xgboostModelWrapper"))

  override protected def create(): XGClassificationModelWrapper = new XGClassificationModelWrapper(dlmc.copy(ParamMap()))

  override def transform(dataset: Dataset[_]): DataFrame = dlmc.transform(dataset)

  override def transformSchema(schema: StructType): StructType = dlmc.transformSchema(schema)

  override def write: WithSummaryWriter[XGClassificationModelWrapper] = new ModelWithSummary.WithSummaryWriter[XGClassificationModelWrapper](this) {
    protected override def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      dlmc.write.save(s"$path/dlmc")
    }
  }
}



object XGClassificationModelWrapper extends MLReadable[XGClassificationModelWrapper] {
  override def read: MLReader[XGClassificationModelWrapper] = new WithSummaryReader[XGClassificationModelWrapper]() {
    override def load(path: String): XGClassificationModelWrapper = {
      super.load(path) match {
        case original: XGClassificationModelWrapper =>
          original._dlmc = DefaultParamsReader.loadParamsInstance(s"$path/dlmc", sc).asInstanceOf[DMLCModel]
          original
      }
    }
  }
}