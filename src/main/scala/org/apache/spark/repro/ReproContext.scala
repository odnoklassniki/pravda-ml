package org.apache.spark.repro

import org.apache.spark.ml.odkl.{HasMetricsBlock, ModelWithSummary}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.{MLWritable, MLWriter}
import org.apache.spark.ml.{Estimator, Model, Pipeline, PipelineModel}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.util.DynamicVariable

trait ReproContext {

  def persistEstimator(estimator: MLWritable): Unit

  def persistModel(model: MLWritable) : Unit

  def dive(tags: Seq[(String, String)]): ReproContext

  def logParams(params: Params, path: Seq[String]): Unit

  def logMetircs(metrics: DataFrame)

  def start(): Unit

  def finish() : Unit
}

trait ParamsHolder {
  def getThis: Option[Params]
  def getNested: Map[String,Params]
}

object ReproContext extends ReproContext with HasMetricsBlock {

  implicit class ReproducibleEstimator[M <:Model[M] with MLWritable](estimator: Estimator[M] with MLWritable) {
    def reproducableFit(dataset: Dataset[_])(implicit initialContext: ReproContext) : M = {

      assert(currentContext.value.isEmpty, "Reproduction context is not empty!")

      currentContext.withValue(List(initialContext)) {
        try {
          initialContext.persistEstimator(estimator)

          logParams(initialContext, Seq(), estimator)

          initialContext.start()

          val model = estimator.fit(dataset)

          logMetrics(initialContext, model)

          initialContext.persistModel(model)

          assert(currentContext.value.size == 1, "Only one context must remain in the stack at the end.")
          assert(currentContext.value.head == initialContext, "The only remaining context should be ours.")

          model
        } finally {
          initialContext.finish()
        }
      }
    }
  }

  private class PhantomWritableEstimator[M <: Model[M]](delegate: Estimator[M]) extends Estimator[M] with MLWritable {
    override def fit(dataset: Dataset[_]): M = delegate.fit(dataset)

    override def copy(extra: ParamMap): Estimator[M] = delegate.copy(extra)

    override def write: MLWriter = new MLWriter {
      override protected def saveImpl(path: String): Unit = {}
    }

    override def transformSchema(schema: StructType): StructType = delegate.transformSchema(schema)

    override val uid: String = delegate.uid
  }

  implicit class TracedEstimator[M <:Model[M] with MLWritable](estimator: Estimator[M]) {
    def tracedFit(dataset: Dataset[_])(implicit initialContext: ReproContext) : M = {
      new PhantomWritableEstimator(estimator).reproducableFit(dataset)
    }
  }

  private val currentContext = new DynamicVariable[List[ReproContext]](List())

  private def logParams(context: ReproContext, path: Seq[String], params: Params) : Unit = {
    params match {
      case pipeline: Pipeline =>
        val stages = pipeline.getStages
        for(i <- stages.indices) logParams(context, path :+ s"stage${i}_${stages(i).getClass.getSimpleName}", stages(i))
      case holder: ParamsHolder =>
        holder.getThis.foreach(x => context.logParams(x, path))
        holder.getNested.foreach(x => logParams(context, path :+ x._1, x._2))
      case _ => context.logParams(params, path)
    }
  }

  private def logMetrics(context: ReproContext, model: Model[_]) : Unit = {
    model match {
      case pipelineMode: PipelineModel =>
        pipelineMode.stages.foreach {
          case nestedModel: Model[_] => logMetrics(context, nestedModel)
          case _ =>
        }
      case withSummary: ModelWithSummary[_] if withSummary.summary.blocks.contains(metrics) =>
        context.logMetircs(withSummary.summary(metrics))
      case _ =>
    }
  }

  override def persistEstimator(estimator: MLWritable): Unit =  currentContext.value.lastOption.foreach(_.persistEstimator(estimator))

  override def persistModel(model: MLWritable): Unit = currentContext.value.lastOption.foreach(_.persistModel(model))

  override def dive(tags: Seq[(String, String)]): ReproContext = {
    val next = currentContext.value.lastOption.map(_.dive(tags))
    next.foreach(x => currentContext.value = currentContext.value :+ x)
    next.getOrElse(this)
  }

  def comeUp(): ReproContext = {
    currentContext.value.lastOption match {
      case Some(current) =>
        current.finish()
        currentContext.value = currentContext.value.dropRight(1)
        assert(currentContext.value.nonEmpty, "There must be at least one more context in the stack.")
        currentContext.value.last
      case _ => this
    }
  }

  override def logParams(params: Params, path: Seq[String]): Unit = currentContext.value.lastOption.foreach(_.logParams(params, path))

  override def logMetircs(metrics: DataFrame): Unit = currentContext.value.lastOption.foreach(_.logMetircs(metrics))

  override def start(): Unit =  ???

  override def finish(): Unit = ???
}