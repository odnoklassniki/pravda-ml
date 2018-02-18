package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utility for splitting fit process into multiple branches,
  * optionally executed in parallel.
  */

import org.apache.commons.lang3.StringUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.ml.util.DefaultParamsReader
import org.apache.spark.sql.{DataFrame, Dataset, functions}
import org.apache.spark.sql.types.StructType

import scala.collection.parallel.TaskSupport

/**
  * Utility used to split training into forks (per type, per class, per fold).
  *
  * @param nested Nested estimator to call for each fork.
  * @tparam ModelIn  Type of model produced by the nested estimator.
  * @tparam ModelOut Type of the resulting model. Does not have to be the same as ModelIn.
  */
abstract class ForkedEstimator[
ModelIn <: ModelWithSummary[ModelIn],
ForeKeyType,
ModelOut <: ModelWithSummary[ModelOut]]
(
  val nested: SummarizableEstimator[ModelIn],
  override val uid: String

)
  extends SummarizableEstimator[ModelOut] with ForkedModelParams {

  final val trainParallel: Param[Boolean] = new Param[Boolean](
    this, "trainParallel", "Whenever to train different parts in parallel")


  final val cacheForks: Param[Boolean] = new Param[Boolean](
    this, "cacheForks", "Useful to reduce IO when training in parallel. If set caches and materializes forks using a single job.")

  final val pathForTempModels = new Param[String](
    this, "pathForTempModels", "Used for incremental training. Persist models when trained and skips training if valid model found.")

  setDefault(trainParallel -> false, cacheForks -> false)

  def setTrainParallel(value: Boolean): this.type = set(trainParallel, value)

  def setCacheForks(value: Boolean): this.type = set(cacheForks, value)

  def setPathForTempModels(value: String): this.type =
    if (StringUtils.isNotBlank(value)) set(pathForTempModels, value) else clear(pathForTempModels)

  /**
    * Override this method and create forks to train from the data.
    */
  protected def createForks(dataset: Dataset[_]): Seq[(ForeKeyType, DataFrame)]

  /**
    * Given models trained for each fork create a combined model. This model is the
    * result of the estimator.
    */
  protected def mergeModels(models: Seq[(ForeKeyType, ModelIn)]): ModelOut

  override def fit(dataset: Dataset[_]): ModelOut = {

    val forks: Seq[(ForeKeyType, DataFrame)] = createForks(dataset)

    if ($(cacheForks)) {
      forks.map(_._2.toDF.cache()).reduce((a, b) => a.union(b)).count()
    }

    try {
      val mayBeParallel = if ($(trainParallel)) {
        val par = forks.par
        par.tasksupport = ForkedEstimator.getTaskSupport
        par
      } else forks


      val models: Array[(ForeKeyType, ModelIn)] = mayBeParallel.map(partialData => try {
        fitFork(dataset, partialData)
      } catch {
        case e: Throwable =>
          logError(s"Exception while handling fork ${partialData._1}", e)
          throw e
      }).toArray

      val result = mergeModels(models).setParent(this)

      if (isDefined(propagatedKeyColumn) && result.isInstanceOf[ForkedModelParams]) {
        result.asInstanceOf[ForkedModelParams].setPropagatedKeyColumn($(propagatedKeyColumn))
      }

      // Result is ready, its time to schedule deletion of save points if enabled.
      if (isDefined(pathForTempModels)) {
        // Do not delete immediately since model files might be needed. Delete only on JVM exit
        FileSystem.get(dataset.sqlContext.sparkContext.hadoopConfiguration).deleteOnExit(new Path($(pathForTempModels)))
      }

      result
    } finally {
      if ($(cacheForks)) {
        forks.foreach(_._2.unpersist())
      }
    }
  }

  def fitFork(wholeData: Dataset[_], partialData: (ForeKeyType, DataFrame)): (ForeKeyType, ModelIn) = {
    logInfo(s"Fitting at $uid for ${partialData._1}...")

    if (isDefined(pathForTempModels)) {
      val path: String = $(pathForTempModels) + s"/key=${partialData._1.toString}"

      if (FileSystem.get(wholeData.sqlContext.sparkContext.hadoopConfiguration).exists(new Path(path))) {
        logInfo(s"At $uid model for ${partialData._1} found in pre-calculated folder $path, loading")
        return (partialData._1, DefaultParamsReader.loadParamsInstance[ModelIn](path, wholeData.sqlContext.sparkContext))
      }
    }

    val result: (ForeKeyType, ModelIn) = (
      partialData._1,
      nested.fit(get(propagatedKeyColumn).map(x => partialData._2.withColumn(x, functions.lit(partialData._1))).getOrElse(partialData._2)))
    logInfo(s"Fitting at $uid for ${partialData._1} DONE")

    if (isDefined(pathForTempModels)) {
      val path: String = $(pathForTempModels) + s"/key=${partialData._1.toString}"
      result._2.write.save(path)
    }
    result
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = nested.transformSchema(schema)
}

/**
  * Helper used to inject common task support with thread count limit into
  * all forked estimators.
  */
object ForkedEstimator extends Serializable {
  private var taskSupport: Option[TaskSupport] = None

  def getTaskSupport = taskSupport.getOrElse(scala.collection.parallel.defaultTaskSupport)

  def setTaskSupport(support: TaskSupport) = taskSupport = Some(support)
}

/**
  * Specific case of forked estimator which does not change the type of the underlying model.
  */
abstract class ForkedEstimatorSameType[
ModelIn <: ModelWithSummary[ModelIn],
ForeKeyType]
(
  nested: SummarizableEstimator[ModelIn],
  override val uid: String
) extends ForkedEstimator[ModelIn, ForeKeyType, ModelIn](nested, uid)


trait ForkedModelParams {
  this: Params =>

  final val propagatedKeyColumn = new Param[String](
    this, "propagatedKeyColumn", "If provided, value of the key the fork is created for is added to the data as a ne column with this name")

  def setPropagatedKeyColumn(value: String): this.type = set(propagatedKeyColumn, value)

  protected def mayBePropagateKey(data: DataFrame, key: Any) : DataFrame = {
    get(propagatedKeyColumn).map(x => data.withColumn(x, functions.lit(key))).getOrElse(data)
  }
}