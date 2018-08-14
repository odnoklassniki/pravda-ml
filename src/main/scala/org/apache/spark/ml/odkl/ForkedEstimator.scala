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

import java.util.concurrent._
import java.util.function.Supplier

import org.apache.commons.lang3.StringUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.DefaultParamsReader
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext, functions}

import scala.collection.parallel.TaskSupport
import scala.util.control.NonFatal
import scala.util.{Success, Try}

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

  final val numThreads = new IntParam(this, "numThreads", "How many threads to use for fitting forks.")


  final val cacheForks: Param[Boolean] = new Param[Boolean](
    this, "cacheForks", "Useful to reduce IO when training in parallel. If set caches and materializes forks using a single job.")

  final val pathForTempModels = new Param[String](
    this, "pathForTempModels", "Used for incremental training. Persist models when trained and skips training if valid model found.")

  final val persistingKeyColumns = new StringArrayParam(this, "persistingKeyColumns",
    "Used to persist resulting model with additional key columns in case if multiple forked estimator are nested")

  final val overwriteModels = new BooleanParam(this, "overwriteModels", "Whenever to allow overwriting models. If not enabled restoration " +
    "after failure might fail for partly written model.")

  setDefault(numThreads -> 1, cacheForks -> false, overwriteModels -> false)

  def setNumThreads(value: Int): this.type = set(numThreads, value)

  def setCacheForks(value: Boolean): this.type = set(cacheForks, value)

  def setPathForTempModels(value: String): this.type =
    if (StringUtils.isNotBlank(value)) set(pathForTempModels, value) else clear(pathForTempModels)

  def setOverwriteModels(value: Boolean) : this.type = set(overwriteModels, value)

  /**
    * Override this method and create forks to train from the data.
    */
  protected def createForks(dataset: Dataset[_]): Seq[(ForeKeyType, DataFrame)]

  /**
    * Given models trained for each fork create a combined model. This model is the
    * result of the estimator.
    */
  protected def mergeModels(sqlContext: SQLContext, models: Seq[(ForeKeyType, Try[ModelIn])]): ModelOut

  override def fit(dataset: Dataset[_]): ModelOut = {

    val forks: Seq[(ForeKeyType, DataFrame)] = createForks(dataset)

    if ($(cacheForks)) {
      forks.map(_._2.toDF.cache()).reduce((a, b) => a.union(b)).count()
    }

    try {

      val currentContext = ForkedEstimator.forkingContext.get()

      val models: Array[(ForeKeyType, Try[ModelIn])] =
        if ($(numThreads) <= 1) {
          forks.map(partialData => {
            fitForkInContext(dataset, currentContext, partialData)
          }).toArray
        }
        else {
          // We are not using default Scala .par feature due to lack of controll for number
          // of running tasks and work stealing feature
          val executor = new ThreadPoolExecutor(
            $(numThreads),
            $(numThreads),
            1, TimeUnit.MINUTES,
            new ArrayBlockingQueue[Runnable](forks.size))

          try {
            val workStealingPreventor = new Semaphore($(numThreads))

            forks
              .map(x => executor.submit(
                new Callable[(ForeKeyType, Try[ModelIn])] {
                  override def call(): (ForeKeyType, Try[ModelIn]) = try {
                    workStealingPreventor.acquire()
                    fitForkInContext(dataset, currentContext, x)
                  } finally {
                    workStealingPreventor.release()
                  }
                }))
              .map(_.get()).toArray
          } finally {
            executor.shutdown()
          }

        }

      val result = mergeModels(dataset.sqlContext, models).setParent(this)

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

  private def fitForkInContext(dataset: Dataset[_], currentContext: Map[String, String], partialData: (ForeKeyType, DataFrame)) = {
    ForkedEstimator.forkingContext.set(currentContext ++ Map(uid -> partialData._1.toString))
    try {
      fitFork(nested, dataset, partialData)
    } finally {
      ForkedEstimator.forkingContext.set(currentContext)
    }
  }

  protected def failFast(key: ForeKeyType, triedIn: Try[ModelIn]): Try[ModelIn] = {
    if (triedIn.isFailure) {
      logError(s"Fitting at $uid failed for $key due to ${triedIn.failed.get}")
    }
    triedIn.get; triedIn
  }

  def fitFork(estimator: SummarizableEstimator[ModelIn], wholeData: Dataset[_], partialData: (ForeKeyType, DataFrame)): (ForeKeyType, Try[ModelIn]) = {
    logInfo(s"Fitting at $uid for ${partialData._1}...")

    val context = ForkedEstimator.forkingContext.get()
    val pathForModel = get(pathForTempModels).map(_ +
      (if(context.nonEmpty) context.values.toArray.sorted.mkString("/context=","_","") else "") +
      s"/key=${partialData._1.toString}")

    if (pathForModel.isDefined) {

      if (FileSystem.get(wholeData.sqlContext.sparkContext.hadoopConfiguration).exists(new Path(pathForModel.get))) {
        logInfo(s"At $uid model for ${partialData._1} found in pre-calculated folder $pathForModel, loading")
        try {
          return (partialData._1, failFast(partialData._1,
            Success(DefaultParamsReader.loadParamsInstance[ModelIn](pathForModel.get, wholeData.sqlContext.sparkContext))))
        } catch {
          case NonFatal(e) => logError(s"Failed to read model from $pathForModel, fitting again.")
        }
      }
    }

    val result: (ForeKeyType, Try[ModelIn]) = (
      partialData._1,
      failFast(partialData._1, Try({
        val model = estimator.fit(get(propagatedKeyColumn).map(x => partialData._2.withColumn(x, functions.lit(partialData._1))).getOrElse(partialData._2))

        logInfo(s"Fitting at $uid for ${partialData._1} DONE")

        if (pathForModel.isDefined) {
          if ($(overwriteModels))
            model.write.overwrite().save(pathForModel.get)
          else
            model.write.save(pathForModel.get)
        }

        model
      })))

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

  private val forkingContext: ThreadLocal[Map[String,String]] = ThreadLocal.withInitial(new Supplier[Map[String, String]] {
    override def get(): Map[String, String] = Map[String,String]()
  })

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