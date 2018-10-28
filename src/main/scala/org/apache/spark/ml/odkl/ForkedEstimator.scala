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

import scala.collection.mutable
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

  final val pathForTempModels = new Param[String](
    this, "pathForTempModels", "Used for incremental training. Persist models when trained and skips training if valid model found.")

  final val persistingKeyColumns = new StringArrayParam(this, "persistingKeyColumns",
    "Used to persist resulting model with additional key columns in case if multiple forked estimator are nested")

  final val overwriteModels = new BooleanParam(this, "overwriteModels", "Whenever to allow overwriting models. If not enabled restoration " +
    "after failure might fail for partly written model.")

  setDefault(numThreads -> 1, overwriteModels -> false)

  def setNumThreads(value: Int): this.type = set(numThreads, value)

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

    val forksSource = createForkSource(dataset)

    try {

      val currentContext = ForkedEstimator.forkingContext.get()

      def fitCycle(): Unit = {
        var nextFork = forksSource.nextFork()
        while (nextFork.isDefined) {
          val (key, result) = fitForkInContext(dataset, currentContext, nextFork.get)
          nextFork = forksSource.consumeFork(key, result)
        }
      }

      val model  =
        if ($(numThreads) <= 1) {

          fitCycle()

          forksSource.createResult()
        }
        else {
          // We are not using default Scala .par feature due to lack of controll for number
          // of running tasks and work stealing feature
          val executor = new ThreadPoolExecutor(
            $(numThreads),
            $(numThreads),
            1, TimeUnit.MINUTES,
            new ArrayBlockingQueue[Runnable]($(numThreads)))

          try {
            val finishIndicator = new Semaphore($(numThreads))
            finishIndicator.acquire($(numThreads))

            Array.tabulate($(numThreads))(_ => new Runnable {
              override def run(): Unit = try {
                fitCycle()
              } finally {
                finishIndicator.release()
              }
            }).foreach(x => executor.submit(x))

            finishIndicator.acquire($(numThreads))

            forksSource.createResult()
          } finally {
            executor.shutdown()
          }
        }

      val result = model.setParent(this)

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
    }
  }

  private def fitForkInContext(dataset: Dataset[_], currentContext: Seq[String], partialData: (ForeKeyType, DataFrame)) = {
    ForkedEstimator.forkingContext.set(currentContext ++ Seq(partialData._1.toString))
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
      context.toArray.mkString("/context=","_",""))

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

  protected def createForkSource(dataset: Dataset[_]) : ForkSource[ModelIn, ForeKeyType, ModelOut] = new ForkSource[ModelIn, ForeKeyType, ModelOut] {
    private val forks: Iterator[(ForeKeyType, DataFrame)] = createForks(dataset).iterator

    override def nextFork(): Option[(ForeKeyType, DataFrame)] = forks.synchronized(
      if (forks.hasNext) Some(forks.next()) else None)

    private val results = mutable.ArrayBuilder.make[(ForeKeyType,Try[ModelIn])]()

    override def consumeFork(key: ForeKeyType, model: Try[ModelIn]): Option[(ForeKeyType, DataFrame)] = {
      val tuple = (key, model)
      results.synchronized(results += tuple)
      nextFork()
    }

    override def createResult(): ModelOut = mergeModels(dataset.sqlContext, results.result())
  }
}

/**
  * Helper used to inject common task support with thread count limit into
  * all forked estimators.
  */
object ForkedEstimator extends Serializable {
  private val forkingContext: ThreadLocal[Seq[String]] = ThreadLocal.withInitial(new Supplier[Seq[String]] {
    override def get(): Seq[String] = Seq[String]()
  })
}

trait ForkSource[
  ModelIn <: ModelWithSummary[ModelIn],
  ForeKeyType,
  ModelOut <: ModelWithSummary[ModelOut]] {

  def nextFork() : Option[(ForeKeyType, DataFrame)]

  def consumeFork(key: ForeKeyType, model : Try[ModelIn]) : Option[(ForeKeyType, DataFrame)]

  def createResult() : ModelOut
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