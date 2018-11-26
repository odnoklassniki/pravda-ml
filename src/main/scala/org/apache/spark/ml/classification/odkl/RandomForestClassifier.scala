package org.apache.spark.ml.classification.odkl

import org.apache.spark.ml.attribute.{AttributeGroup, BinaryAttribute, NominalAttribute}
import org.apache.spark.ml.{PipelineModel, PredictorParams}
import org.apache.spark.ml.classification.{ProbabilisticClassifierParams, RandomForestClassificationModel}
import org.apache.spark.ml.odkl.ModelWithSummary.{Block, WithSummaryReader, WithSummaryWriter}
import org.apache.spark.ml.odkl.{HasFeaturesSignificance, ModelWithSummary, SummarizableEstimator}
import org.apache.spark.ml.param.{BooleanParam, ParamMap, Params}
import org.apache.spark.ml.tree.RandomForestClassifierParams
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.classification.{RandomForestClassifier => SparkRandomForestEstimator}


class RandomForestClassifier(override val uid: String) extends SummarizableEstimator[RfClassificationModelWrapper]
  with ProbabilisticClassifierParams with DefaultParamsWritable with RandomForestClassifierParams with HasFeaturesSignificance
{

  val addSignificance = new BooleanParam(this, "addSignificance",
  "Whenever to add feature significance block to model summary.")

  setDefault(
    addSignificance -> true)

  def this() = this(Identifiable.randomUID("rfEstimatorWrapper"))

  override def setMaxDepth(value: Int): this.type = set(maxDepth, value)
  override def setMaxBins(value: Int): this.type = set(maxBins, value)
  override def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)
  override def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)
  override def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)
  override def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)
  override def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)
  override def setImpurity(value: String): this.type = set(impurity, value)
  override def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)
  override def setSeed(value: Long): this.type = set(seed, value)
  override def setNumTrees(value: Int): this.type = set(numTrees, value)
  override def setFeatureSubsetStrategy(value: String): this.type =
    set(featureSubsetStrategy, value)

   def setAddSignificance(value: Boolean): this.type = set(addSignificance, value)

  def setFeatureCol(value: String): this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)


  override def copy(extra: ParamMap): SummarizableEstimator[RfClassificationModelWrapper] = defaultCopy(extra)

  private def trainInternal(dataset: Dataset[_]): RandomForestClassificationModel = {
    val estimator = copyValues(new SparkRandomForestEstimator())

    var blocks = Map[Block, DataFrame]()

    estimator.fit(dataset)

  }

  override def fit(dataset: Dataset[_]): RfClassificationModelWrapper = {

    val classificationModelWrapper = new RfClassificationModelWrapper(trainInternal(dataset))

    var blocks = Map[Block, DataFrame]()
    val sqlc = dataset.sqlContext
    val sc = sqlc.sparkContext

    import sqlc.implicits._


    if ($(addSignificance)){
      val features: Array[(Int, String, String)] = AttributeGroup.fromStructField(dataset.schema(getFeaturesCol)).attributes
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
        })).get


      val featureImportance = (classificationModelWrapper.rfModel.featureImportances.toArray zip features)
          .map{case (sig, feature) => (feature._1,feature._3, sig, feature._2)}


      blocks += featuresSignificance -> {sc.parallelize(featureImportance, 1).toDF("idx", "flag", "importance", "name")}
    }

    copyValues(classificationModelWrapper.copy(blocks).setParent(this))

  }

  override def transformSchema(schema: StructType): StructType = copyValues(new SparkRandomForestEstimator()).transformSchema(schema)
}

object RandomForestClassifier extends DefaultParamsReadable[RandomForestClassifier]


class RfClassificationModelWrapper(private var _rfModel: RandomForestClassificationModel, override val uid: String)
  extends ModelWithSummary[RfClassificationModelWrapper]
  with PredictorParams with RandomForestClassifierParams {

  def rfModel: RandomForestClassificationModel = this._rfModel

  /**
    * For serialization only
    */
  @Deprecated def this(uid: String) = this(null, uid)

  def this(rf: RandomForestClassificationModel) = this(rf, Identifiable.randomUID("rfModelWrapper"))

  override protected def create(): RfClassificationModelWrapper = new RfClassificationModelWrapper(rfModel.copy(ParamMap()))

  override def transform(dataset: Dataset[_]): DataFrame = {
    rfModel.transform(dataset)
  }

  override def transformSchema(schema: StructType): StructType = rfModel.transformSchema(schema)

  override def write: WithSummaryWriter[RfClassificationModelWrapper] = new ModelWithSummary.WithSummaryWriter[RfClassificationModelWrapper](this) {
    protected override def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      rfModel.write.save(s"$path/rf")
    }
  }
}



object RfClassificationModelWrapper extends MLReadable[RfClassificationModelWrapper] {
  override def read: MLReader[RfClassificationModelWrapper] = new WithSummaryReader[RfClassificationModelWrapper]() {
    override def load(path: String): RfClassificationModelWrapper = {
      super.load(path) match {
        case original: RfClassificationModelWrapper =>
          original._rfModel = DefaultParamsReader.loadParamsInstance(s"$path/rf", sc).asInstanceOf[RandomForestClassificationModel]
          original
      }
    }
  }
}