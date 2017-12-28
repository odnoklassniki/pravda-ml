package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains common traits and classes for training models
  * with summary.
  */

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml._
import org.apache.spark.ml.odkl.ModelWithSummary.{Block, WithSummaryReaderUntyped, WithSummaryWriter}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, ParamPair}
import org.apache.spark.ml.util._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, functions}
import org.json4s.DefaultWriters._
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, JValue}

/**
  * Model which has a summary. Includes support for reading and wirting summary blocks.
  */
trait ModelWithSummary[M <: ModelWithSummary[M]]
  extends Model[M] with MLWritable   {

  protected val summaryParam =
    new Param[ModelSummary](this.uid, "summary", "Collection of named dataframes with extended information about the model") {
      override def jsonEncode(value: ModelSummary): String = {
        val jValue: JValue = JsonMethods.render(JsonMethods.asJValue(value.blocks.map(x => x._1.name).toArray))
        JsonMethods.compact(jValue)
      }

      override def jsonDecode(json: String): ModelSummary = {

        implicit val formats = DefaultFormats
        val blocks = JsonMethods.parse(json).extract[Array[String]].map(
          x => Block(x) -> null).toMap.asInstanceOf[Map[Block, DataFrame]]

        new ModelSummary(blocks)
      }
    }

  private val saveSummaryParam = new BooleanParam(this, "saveSumary", "Used to disable svaing summaries for nested models")

  setDefault(summaryParam -> new ModelSummary(Map()), saveSummaryParam -> true)

  private[odkl] def setSummary(summary : ModelSummary) : M = set(summaryParam, summary).asInstanceOf[M]

  def disableSaveSummary() = set(saveSummaryParam, false)

  def isSaveSummaryEnabled = $(saveSummaryParam)

  def summary = $(summaryParam)

  override def copy(extra: ParamMap): M = copyValues(create(), extra)

  def copy(blocks: Map[Block, DataFrame], params: ParamMap = ParamMap()): M = {
    copy(params).setSummary(summary.copy(blocks))
  }

  def copy(summary: ModelSummary, params: ParamMap): M =
    copy(summary.blocks, params)

  protected def create(): M

  override def write: ModelWithSummary.WithSummaryWriter[M] = new ModelWithSummary.WithSummaryWriter[M](this)
}

/**
  * One of main extensions to the base concept of model - each model might return a summary represented by a named
  * collection of dataframes.
  *
  * @param blocks Named collection of the dataframe.
  */
class ModelSummary(@transient val blocks: Map[Block, DataFrame] = Map()) extends Serializable {

  def transform(transformer: (Block, (DataFrame) => DataFrame), additional: (Block, (DataFrame) => DataFrame)*): ModelSummary = {
    val transformers = (Seq(transformer) ++ additional).toMap
    val identity = (y: DataFrame) => y

    copy(blocks.map(x => x._1 -> transformers.getOrElse(x._1, identity).apply(x._2)))
  }

  def copy(blocks: Map[Block, DataFrame]): ModelSummary = new ModelSummary(this.blocks ++ blocks)

  def $(block: Block): DataFrame = blocks(block)

  def apply(block: Block): DataFrame = blocks(block)
}

/**
  * Helper for reading and writing models in a typed way.
  */
object ModelWithSummary extends MLReadable[PipelineStage] with Serializable {

  /**
    * Case class used to identify parts in summary (somewhat similar to Param)
    *
    * @param name Name of the block.
    */
  case class Block(val name: String)

  def reader[M <: ModelWithSummary[M]](clazz: Class[_ <: ModelWithSummary[M]]): WithSummaryReader[M]
  = new WithSummaryReader[M]

  def extractSummary(pipelineModel: PipelineModel): ModelSummary = {
    val markedBlocks: Array[(Block, DataFrame)] = pipelineModel.stages.flatMap {
      case model: ModelWithSummary[_] =>
        model.summary.blocks.mapValues(_.withColumn("stageId", functions.lit(model.uid)))
      case _ => Seq()
    }
    val blocks: Map[Block, DataFrame] = markedBlocks.groupBy(_._1)
      .transform((block, data) => data.map(_._2).reduce((a, b) => a.unionAll(b)))

    new ModelSummary(blocks)
  }

  @Since("1.6.0")
  override def read: MLReader[PipelineStage] = new WithSummaryReaderUntyped

  /**
    * Writes model with its summary blocks. Each block is saved into a dedicated folder.
    *
    * @param instance Model to write.
    */
  class WithSummaryWriter[M <: ModelWithSummary[M]]
  (instance: ModelWithSummary[M])
    extends DefaultParamsWriter(instance) {

    protected override def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      if (instance.isSaveSummaryEnabled) {
        instance.summary.blocks.foreach(x => {
          x._2.repartition(1).write.parquet(new Path(path, x._1.name).toString)
        })
      }
    }
  }


  /**
    * Reads model with its summary blocks.
    *
    * @tparam M Type of the model to read.
    */
  class WithSummaryReader[M <: ModelWithSummary[M]] extends MLReader[M] {
    override def load(path: String): M = {
      new WithSummaryReaderUntyped().load(path).asInstanceOf[M]
    }
  }

  /**
    * Reads model with its summary blocks. Does not require certain type (used to read
    * combined models).
    */
  class WithSummaryReaderUntyped
    extends DefaultParamsReader[PipelineStage] {

    override def load(path: String): PipelineStage = {
      super.load(path) match {
        case model: ModelWithSummary[_] =>
          val summary = model.summary
          if (model.isSaveSummaryEnabled) {
            val blocks = summary.blocks.map(x => x._1 -> sqlContext.read.parquet(new Path(path, x._1.name).toString))
            model.set(model.summaryParam, summary.copy(blocks))
          }
          model
      }
    }
  }
}

/**
  * Estimator with produces model with summary. Used to simplify chaining.
  */
trait SummarizableEstimator[M <: ModelWithSummary[M]] extends Estimator[M] {
  override def copy(extra: ParamMap): SummarizableEstimator[M]
}

/**
  * Utility used to bridge default spark ML models into our advanced pipelines.
  * TODO: Provide summary extractors
  */
class MLWrapper[M <: Model[M]](val nested: Estimator[M],
                               val summaryExtractor: M => Map[Block,DataFrame],
                               override val uid: String) extends SummarizableEstimator[MLWrapperModel[M]] {

  def this(nested: Estimator[M], summaryExtractor: M => Map[Block,DataFrame]) =
    this(nested, summaryExtractor, Identifiable.randomUID("wrapperEstimator"))

  def this(nested: Estimator[M]) = this(nested, (x: M) => Map())

  override def fit(dataset: DataFrame): MLWrapperModel[M] = {
    val model = nested.fit(dataset)
    new MLWrapperModel[M](model).copy(summaryExtractor(model)).setParent(this)
  }

  override def transformSchema(schema: StructType): StructType = nested.transformSchema(schema)

  override def copy(extra: ParamMap): MLWrapper[M] = new MLWrapper[M](nested.copy(extra))
}

class MLWrapperModel[M <: Model[M]](val nestedModel: M,
                                    override val uid: String)
  extends ModelWithSummary[MLWrapperModel[M]] {

  private var nestedModelRef = nestedModel

  def this(uid: String) = this(null.asInstanceOf[M], uid)

  def this(nested: M) = this(nested, Identifiable.randomUID("wrapper"))

  def nested: M = nestedModelRef

  override def transform(dataset: DataFrame): DataFrame = nested.transform(dataset)

  override def transformSchema(schema: StructType): StructType = nested.transformSchema(schema)

  override protected def create(): MLWrapperModel[M] = ???

  override def copy(extra: ParamMap): MLWrapperModel[M] = new MLWrapperModel[M](nestedModel.copy(extra))

  override def write: WithSummaryWriter[MLWrapperModel[M]] = new ModelWithSummary.WithSummaryWriter[MLWrapperModel[M]](this) {
    protected override def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      nested match {
        case x : MLWritable => x.write.save(s"$path/nested")
        case _ => throw new UnsupportedOperationException("Can not write nested model of type ")
      }
    }
  }
}

object MLWrapperModel extends MLReadable[PipelineStage] {
  override def read: MLReader[PipelineStage] = new WithSummaryReaderUntyped() {
    override def load(path: String): PipelineStage = {
      super.load(path) match {
        case original : MLWrapperModel[_] =>
          original.nestedModelRef = DefaultParamsReader.loadParamsInstance(s"$path/nested", sc)
          original
      }
    }
  }
}




