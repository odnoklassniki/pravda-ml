package org.apache.spark.ml.odkl.hyperopt

import org.apache.spark.ml.odkl.{ForkSource, ForkedEstimator, ModelWithSummary, SummarizableEstimator}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext, functions}

import scala.collection.mutable
import scala.util.Try

/**
  * Utility used to perform stepwise search for hyper-parameters. Usefull in case if there are certain groups of
  * parameters which do not influence each other and can be optimized separatelly. Pass sequence of optimizers
  * to the grouped search to apply sequential optimization.
  *
  * NB: Grouped search must itself be configured to use single thread, but nested optimizers allowed to use as
  * many threads as they need.
  */
class GroupedSearch[ModelIn <: ModelWithSummary[ModelIn]]
(
  nested: Seq[(String,HyperparametersOptimizer[ModelIn])],
  override val uid: String) extends ForkedEstimator[ModelIn, OptimizerStage, ModelIn](nested.head._2, uid)
  with HasConfigurations {

  val stageNameColumnName = new Param[String](this, "stageNameColumnName",
    "Name of the column in summary blocks to store stage name.")

  val stageReversedIndexColumnName = new Param[String](this, "stageReversedIndexColumnName",
    "Name of the column in summary blocks to store stage reversed index. The last stage will always " +
      "have reversed index equals 0 simplifying extraction of the final result.")


  setDefault(
    stageNameColumnName -> "stageName",
    stageReversedIndexColumnName -> "stageReversedIndex"
  )

  def this(nested: Seq[(String,HyperparametersOptimizer[ModelIn])]) = this(nested, Identifiable.randomUID("groupedOptimizer"))


  override def fit(dataset: Dataset[_]): ModelIn = {
    require($(numThreads) == 1, "Grouped optimization must not be performed in parallel threads")
    super.fit(dataset)
  }


  override def fitFork(estimator: SummarizableEstimator[ModelIn], wholeData: Dataset[_], partialData: (OptimizerStage, DataFrame)): (OptimizerStage, Try[ModelIn]) = {
    val estimatorCopy = nested.find(_._1 == partialData._1.stage).get._2.copy(partialData._1.accumulated)

    super.fitFork(estimatorCopy, wholeData, partialData)
  }

  override protected def createForkSource(dataset: Dataset[_]): ForkSource[ModelIn, OptimizerStage, ModelIn] = {
    new ForkSource[ModelIn, OptimizerStage, ModelIn] {

      override def nextFork(): Option[(OptimizerStage, DataFrame)] =
        Some(OptimizerStage(nested.head._1, ParamMap()), dataset.toDF())

      private val results = mutable.ArrayBuffer[(String, ModelIn, ParamMap)]()

      override def consumeFork(key: OptimizerStage, model: Try[ModelIn]): Option[(OptimizerStage, DataFrame)] = {
        val index = nested.indexWhere(_._1 == key.stage)

        require(index >= 0, s"Unrecognized optimizer stage name ${key.stage}")

        val currentModel = model.get

        val (_, config: ParamMap) = nested(index)._2.extractConfig(currentModel)

        val filledConfig = results.foldLeft(currentModel.summary(configurations))(
          (data, previous) => {
            val estimator = nested.find(_._1 == previous._1).get._2

            previous._3.toSeq.foldLeft(data)((data, param) => data.withColumn(
              estimator.resolveParamName(param.param),
              functions.lit(param.value)
            ))
          })

        results += Tuple3(key.stage, currentModel.copy(Map(configurations -> filledConfig)), config)

        if (index < nested.size - 1) {
          Some(OptimizerStage(nested(index + 1)._1, key.accumulated ++ config), dataset.toDF())
        } else {
          None
        }
      }

      override def createResult(): ModelIn = {
        val blocks = results.head._2.summary.blocks.keys

        val transformedBlocks = blocks.map(block => block -> results.zipWithIndex.map(model => model._1._2.summary.blocks(block).withColumns(
          Seq($(stageNameColumnName), $(stageReversedIndexColumnName)),
          Seq(functions.lit(model._1._1), functions.lit(results.size - model._2 - 1))
        )).reduce((overall, current) => {
          val missingInOveral = current.schema.fieldNames.toSet.diff(overall.schema.fieldNames.toSet)
          val missingInCurrent = overall.schema.fieldNames.toSet.diff(current.schema.fieldNames.toSet)

          val patchedOveral = missingInOveral.foldLeft(overall)(
            (data, column) => data.withColumn(column, functions.lit(null).cast(current.schema(column).dataType)))

          val patchedCurrent = missingInCurrent.foldLeft(current)(
            (data, column) => data.withColumn(column, functions.lit(null).cast(overall.schema(column).dataType)))

          patchedOveral.unionByName(patchedCurrent)
        }))
          .toMap

        results.last._2.copy(transformedBlocks)
      }
    }
  }

  override def copy(extra: ParamMap): GroupedSearch[ModelIn] = copyValues(new GroupedSearch[ModelIn](
    nested.map(x => x._1 -> x._2.copy(extra))))

  /**
    * Override this method and create forks to train from the data.
    */
  override protected def createForks(dataset: Dataset[_]): Seq[(OptimizerStage, DataFrame)] = ???

  /**
    * Given models trained for each fork create a combined model. This model is the
    * result of the estimator.
    */
  override protected def mergeModels(sqlContext: SQLContext, models: Seq[(OptimizerStage, Try[ModelIn])]): ModelIn = ???
}

case class OptimizerStage(stage: String, accumulated: ParamMap) {
  override def toString: String = s"stage_$stage"
}