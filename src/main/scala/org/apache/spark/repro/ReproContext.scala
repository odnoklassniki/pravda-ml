package org.apache.spark.repro

import org.apache.spark.ml.odkl.UnwrappedStage.{DynamicDataTransformerTrainer, IdentityModelTransformer, NoTrainEstimator}
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.param.{Param, ParamMap, ParamPair, Params}
import org.apache.spark.ml.util.{MLWritable, MLWriter}
import org.apache.spark.ml._
import org.apache.spark.ml.odkl.Evaluator.EvaluatingTransformer
import org.apache.spark.ml.odkl.hyperopt.HasConfigurations
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, functions}

import scala.annotation.tailrec
import scala.util.DynamicVariable

trait ReproContext {

  def persistEstimator(estimator: MLWritable): Unit

  def persistModel(model: MLWritable) : Unit

  def dive(tags: Seq[(String, String)]): ReproContext

  def logParamPairs(params: Iterable[ParamPair[_]], path: Seq[String]): Unit

  def logParams(params: Params, path: Seq[String]): Unit = {
    val pairs = params
      .params.view
      .filter(x => params.get(x).exists(y => params.getDefault(x).isEmpty || y != params.getDefault(x).get))
      .map(x => ParamPair[Any](x.asInstanceOf[Param[Any]], params.get(x).get))

    logParamPairs(pairs, path)
  }

  def logMetircs(metrics: => DataFrame)

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

  private class PhantomWritableEstimator[M <: Model[M]](val delegate: Estimator[M]) extends Estimator[M] with MLWritable {
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
      case phantom: PhantomWritableEstimator[_] => logParams(context, path, phantom.delegate)
      case pipeline: Pipeline =>
        val stages = pipeline.getStages
        for(i <- stages.indices) logParams(context, path :+ s"stage${i}_${getNameToLog(stages(i))}", stages(i))
      case holder: ParamsHolder =>
        holder.getThis.foreach(x => context.logParams(x, path))
        holder.getNested.foreach(x => logParams(context, path :+ x._1, x._2))
      case unwrapped: UnwrappedStage[_, _] =>
        val estimator = unwrapped.estimator
        logParams(context, path :+ s"${getNameToLog(estimator)}", estimator)

        unwrapped.transformerTrainer match {
          case noTrain : NoTrainEstimator[_, _] =>
            noTrain.transformer match {
              case identity: IdentityModelTransformer[_] => logParams(context, path, identity.dataTransformer)
              case _ => logParams(context, path, noTrain.transformer)
            }
          case dynamicData : DynamicDataTransformerTrainer[_] =>
            logParams(context, path, dynamicData.nested)
          case transformerTrainer: Estimator[_] =>
            logParams(context, path, transformerTrainer)
        }

      case evaluator: EvaluatingTransformer[_, _] =>
        logParams(context, path :+ s"${getNameToLog(evaluator.evaluator)}", evaluator.evaluator)

      case forked: ForkedEstimator[_, _, _]  =>
        context.logParams(forked, path)
        logParams(context, path :+ s"${getNameToLog(forked.nested)}", forked.nested)

      case _ => context.logParams(params, path)
    }
  }

  @tailrec
  private def getNameToLog(estimator: PipelineStage) : String = {
    estimator match {
      case unwrapped: UnwrappedStage[_,_] => getNameToLog(unwrapped.transformerTrainer)
      case noTrain : NoTrainEstimator[_, _] => noTrain.transformer match {
        case identity: IdentityModelTransformer[_] => getNameToLog(identity.dataTransformer)
        case _ => getNameToLog(noTrain.transformer)
      }
      case _ => estimator.getClass.getSimpleName
    }
  }

  private def logMetrics(context: ReproContext, model: Model[_]) : Unit = {
    model match {
      case pipelineMode: PipelineModel =>
        pipelineMode.stages.foreach {
          case nestedModel: Model[_] => logMetrics(context, nestedModel)
          case _ =>
        }
      case withSummary: ModelWithSummary[_] =>
        (withSummary.parent match {
          case extractor: MetricsExtractor =>
            extractor.extract(withSummary)
          case _ =>
            withSummary.summary.blocks.get(metrics)

        }).foreach(x => logMetircs(x))

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

  override def logParamPairs(params: Iterable[ParamPair[_]], path: Seq[String]): Unit = currentContext.value.lastOption.foreach(_.logParamPairs(params, path))

  override def logMetircs(metrics: => DataFrame): Unit = currentContext.value.lastOption.foreach(_.logMetircs(metrics))

  def logMetricsFromModel(model : Model[_]) : Unit = {
    currentContext.value.lastOption.foreach(x => logMetrics(x, model))
  }

  override def start(): Unit =  ???

  override def finish(): Unit = ???
}