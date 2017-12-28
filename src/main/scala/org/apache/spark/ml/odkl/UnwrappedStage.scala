package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utility for injection of train-only stages into pipeline. These stages
  * are applied only while fitting the model and do not apear in the result (not applied when predicting).
  */

import java.util.concurrent.ThreadLocalRandom

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.sql.catalyst.expressions.Literal
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Column, DataFrame, Row, functions}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable

/**
  * In case if we can avoid certain stages used during training while predicting we need to propagate
  * some changes to the model (eg. unscale weights or remove intercept). Also useful for extending summary
  * blocks (eg. during evaluation/cross-validation).
  *
  * This interface defines the logic of model transformation.
  */
trait ModelTransformer[M <: ModelWithSummary[M], T <: ModelTransformer[M, T]] extends Model[T] with Logging {
  def transformModel(model: M, originalData: DataFrame): M

  def copy(extra: ParamMap): T = defaultCopy(extra)

  def release(originalData: DataFrame, transformedData: DataFrame) = {}
}

/**
  * In case if we can avoid certain stages used during training while predicting we need to propagate
  * some changes to the model (eg. unscale weights or remove intercept). Also useful for extending summary
  * blocks (eg. during evaluation/cross-validation).
  *
  * This class is used as a typical pipeline stage while training (fits and applies transformer, then calls the
  * nested estimator), but it automatically eliminates itself from the resulting model by applying model transformer.
  *
  * @param estimator          Nested estimator to pass control
  * @param transformerTrainer Estimator for fitting data AND model transformer.
  */
class UnwrappedStage[M <: ModelWithSummary[M], T <: ModelTransformer[M, T]]
(
  estimator: Estimator[M],
  transformerTrainer: Estimator[T],
  override val uid: String) extends SummarizableEstimator[M] {

  def cacheTransformed = new BooleanParam(
    this, "cacheTransformed", "Whenever to cache data returned by the transformer. If multiple " +
      "elliminatable transformations are pipelined, it might be worth to cache results on the top level")

  def materializeCached = new BooleanParam(
    this, "materializeCached", "Whenever to materialize cached data. If nested estimator is parallelizable it is " +
      "worth doing. Otherwise cached data might be materialized more than once.")

  setDefault(cacheTransformed -> false, materializeCached -> false)

  def setCacheTransformed(value: Boolean = true): this.type = set(cacheTransformed, value)

  def setMaterializeCached(value: Boolean = true): this.type = set(materializeCached, value)

  def this(estimator: Estimator[M], transformerTrainer: Estimator[T]) =
    this(estimator, transformerTrainer, Identifiable.randomUID("unwrappedPipeline"))

  def this(estimator: Estimator[M], transformer: T) =
    this(estimator, new UnwrappedStage.NoTrainEstimator[M, T](transformer))

  override def fit(dataset: DataFrame): M = {

    val transformer = transformerTrainer.fit(dataset)

    val transformed = transformer.transform(dataset)

    val mayBeCached = if ($(cacheTransformed)) {
      val data = transformed.cache()
      if ($(materializeCached)) {
        data.count()
      }

      data
    } else {
      transformed
    }

    try {
      val model: M = estimator.fit(mayBeCached)

      transformer.transformModel(model, dataset)
    } finally {
      transformer.release(originalData = dataset, transformedData = transformed)
      if ($(cacheTransformed)) {
        mayBeCached.unpersist()
      }
    }
  }

  override def copy(extra: ParamMap): SummarizableEstimator[M] = copyValues(new UnwrappedStage[M, T](
    estimator.copy(extra),
    transformerTrainer.copy(extra)))

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = transformerTrainer.transformSchema(schema)
}

object UnwrappedStage extends Serializable {

  /**
    * Adds a stage with data downstream transformation and model upstream transformation.
    */
  def wrap[M <: ModelWithSummary[M], T <: ModelTransformer[M, T]]
  (estimator: SummarizableEstimator[M], unwrapableEstimator: Estimator[T])
  = new UnwrappedStage[M, T](estimator, unwrapableEstimator)

  /**
    * Adds a stage with model only transformation (eg. evaluation)
    */
  def modelOnly[M <: ModelWithSummary[M], T <: ModelTransformer[M, T]]
  (estimator: SummarizableEstimator[M], modelTransformer: T)
  = new UnwrappedStage[M, T](
    estimator, new NoTrainEstimator[M, T](modelTransformer))

  /**
    * Adds a stage with data-only transformation (eg. assigning folds).
    */
  def dataOnly[M <: ModelWithSummary[M]]
  (
    estimator: SummarizableEstimator[M],
    dataTransformer: Transformer)
  = new UnwrappedStage[M, IdentityModelTransformer[M]](
    estimator, new NoTrainEstimator[M, IdentityModelTransformer[M]](new IdentityModelTransformer[M](dataTransformer)))

  /**
    * Adds a stage with data-only transformation (eg. assigning folds).
    */
  def dataOnlyWithTraining[M <: ModelWithSummary[M]]
  (
    estimator: SummarizableEstimator[M],
    dataTransformerFitter: Estimator[_])
  = new UnwrappedStage[M, IdentityModelTransformer[M]](
    estimator, new DynamicDataTransformerTrainer[M](dataTransformerFitter))

  /**
    * Cache data before passing to estimator (won't be cached in resulting prediction model).
    */
  def cache[M <: ModelWithSummary[M]](estimator: SummarizableEstimator[M], storageLevel: StorageLevel = StorageLevel.MEMORY_ONLY): UnwrappedStage[M, CachingTransformer[M]] = {
    new UnwrappedStage[M, CachingTransformer[M]](estimator, new CachingTransformer[M]().setStorageLevel(storageLevel))
  }

  /**
    * Cache data before passing to estimator (won't be cached in resulting prediction model). Forces cache materialization
    * by calling count.
    */
  def cacheAndMaterialize[M <: ModelWithSummary[M]](estimator: SummarizableEstimator[M], storageLevel: StorageLevel = StorageLevel.MEMORY_ONLY): UnwrappedStage[M, CachingTransformer[M]] = {
    new UnwrappedStage[M, CachingTransformer[M]](estimator, new CachingTransformer[M]().setMaterializeCached().setStorageLevel(storageLevel))
  }

  /**
    * Cache data before passing to estimator (won't be cached in resulting prediction model).
    */
  def cache[M <: ModelWithSummary[M]](estimator: SummarizableEstimator[M], cacher: CachingTransformer[M]): UnwrappedStage[M, CachingTransformer[M]] = {
    new UnwrappedStage[M, CachingTransformer[M]](estimator, cacher)
  }

  /**
    * Stores data into temporary path. Usefull for "grounding" data and avoiding large execution plans.
    */
  def persistToTemp[M <: ModelWithSummary[M]](
                                               estimator: SummarizableEstimator[M],
                                               tempPath: String,
                                               uncacheInput: Boolean = false,
                                               partitionBy: Array[String] = Array()): UnwrappedStage[M, PersistingTransformer[M]] = {
    new UnwrappedStage[M, PersistingTransformer[M]](
      estimator, new PersistingTransformer[M]().setTempPath(tempPath).setPartitionBy(partitionBy))
  }

  /**
    * Repartition the data before passing to estimator. Reparitioning will not apear in the resulting prediction model.
    *
    * @param estimator     Estimator to add partitioning to.
    * @param numPartitions Number of partitions.
    * @param partitionBy   Columns to partition by.
    * @param sortBy        Columns to sort data in partitions. Note that partitionBy are not added to this set by default.
    * @return Exactly the same model as produced by the estimator.
    */
  def repartition[M <: ModelWithSummary[M]]
  (
    estimator: SummarizableEstimator[M],
    numPartitions: Int,
    partitionBy: Seq[String],
    sortBy: Seq[String]): UnwrappedStage[M, IdentityModelTransformer[M]] = {
    val partitioner: PartitioningTransformer = new PartitioningTransformer()
    repartition(
      estimator,
      partitioner
        .set(partitioner.numPartitions, numPartitions)
        .set(partitioner.partitionBy, partitionBy.toArray)
        .set(partitioner.sortBy, sortBy.toArray))
  }

  /**
    * Repartition the data before passing to estimator. Reparitioning will not apear in the resulting prediction model.
    *
    * @param partitioner Defines the logic of partitioning.
    */
  def repartition[M <: ModelWithSummary[M]](estimator: SummarizableEstimator[M], partitioner: PartitioningTransformer): UnwrappedStage[M, IdentityModelTransformer[M]] = {
    dataOnly(estimator, partitioner)
  }

  /**
    * Repartition the data before passing to estimator. Reparitioning will not apear in the resulting prediction model.
    *
    * @param estimator     Estimator to add partitioning to.
    * @param numPartitions Number of partitions.
    * @return Exactly the same model as produced by the estimator.
    */
  def repartition[M <: ModelWithSummary[M]]
  (
    estimator: SummarizableEstimator[M],
    numPartitions: Int): UnwrappedStage[M, IdentityModelTransformer[M]] = {
    repartition(estimator, numPartitions, Array[String](), Array[String]())
  }

  /**
    * Repartition the data before passing to estimator. Reparitioning will not apear in the resulting prediction model.
    *
    * @param estimator     Estimator to add partitioning to.
    * @param numPartitions Number of partitions.
    * @param partitionBy   Columns to partition by.
    * @return Exactly the same model as produced by the estimator.
    */
  def repartition[M <: ModelWithSummary[M]]
  (
    estimator: SummarizableEstimator[M],
    numPartitions: Int,
    partitionBy: Seq[String]): UnwrappedStage[M, IdentityModelTransformer[M]] = {
    repartition(estimator, numPartitions, partitionBy, Array[String]())
  }

  /**
    * Keeps only predefined set of columns in the dataset before passing to estimator. Usefull in combination
    * with caching to reduce memory footprint. Projection will not appear in the resulting prediction model.
    *
    * @param estimator Estimator to cal after projecting.
    * @param columns   Columns to keep.
    * @return Exactly the same model as produced by the estimator.
    */
  def project[M <: ModelWithSummary[M]](estimator: SummarizableEstimator[M], columns: Seq[String]): UnwrappedStage[M, IdentityModelTransformer[M]] = {
    dataOnly(estimator, new ProjectingTransformer().setColumnsToKeep(columns))
  }

  /**
    * Removes predefined set of columns in the dataset before passing to estimator. Usefull in combination
    * with caching to reduce memory footprint. Projection will not appear in the resulting prediction model.
    *
    * @param estimator Estimator to cal after projecting.
    * @param columns   Columns to remove.
    * @return Exactly the same model as produced by the estimator.
    */
  def projectInverse[M <: ModelWithSummary[M]](estimator: SummarizableEstimator[M], columns: Seq[String]): UnwrappedStage[M, IdentityModelTransformer[M]] = {
    dataOnly(estimator, new ProjectingTransformer().setColumnsToRemove(columns))
  }

  /**
    * Collect all summary blocks to driver and add re-create dataframe with a single block. Usefull to reduce number
    * of partitions and tasks for the final persist.
    *
    * @param estimator Estimator to wrap summary blocks for.
    * @return Final model is the same, but summary blocks are collected and re-created.
    */
  def collectSummary[M <: ModelWithSummary[M]](estimator: SummarizableEstimator[M]): UnwrappedStage[M, CollectSummaryTransformer[M]] = {
    modelOnly(estimator, new CollectSummaryTransformer[M]())
  }

  /**
    * Adds a stage for sampling data from the dataset. Behavior is deterministic (iteration always produce
    * the same result) if withReplacement OR seed specified, otherwise the behavior is non-determenistic
    * and subsequent iterations migth see different samples.
    * @param estimator Estimator to sample data for.
    * @param numRecords Expected number of records to sample
    * @param withReplacement Whenever to simulate replacement (single item might be selected multiple times)
    * @param seed Seed for the random number generation.
    * @return Estimator with samples data before passing to nested estimator.
    */
  def sample[M <: ModelWithSummary[M]]
  (
    estimator: SummarizableEstimator[M],
    numRecords: Int,
    withReplacement: Boolean = false,
    seed: Option[Long] = None): UnwrappedStage[M, IdentityModelTransformer[M]] = {

    val trainer = new DynamicDownsamplerTrainer().setExpectedRecords(numRecords)
    trainer.set(trainer.withReplacement, withReplacement)
    seed.foreach(x => trainer.set(trainer.seed, x))
    dataOnlyWithTraining(estimator, trainer)
  }

  /**
    * Utility simplifying creation of predefined model transformer (when no fitting required).
    *
    * @param transformer Transformer to return.
    */
  class NoTrainEstimator[M <: ModelWithSummary[M], T <: ModelTransformer[M, T]]
  (
    override val uid: String,
    transformer: T)
    extends Estimator[T] with DefaultParamsWritable {

    def this(transformer: T) = this(Identifiable.randomUID("noTrainEstimator"), transformer)

    override def fit(dataset: DataFrame): T = transformer

    override def copy(extra: ParamMap): Estimator[T]
    = copyValues(new NoTrainEstimator[M, T](transformer))

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema
  }

  /**
    * Utility simplifying transformations when only model transformation is required.
    */
  abstract class ModelOnlyTransformer[M <: ModelWithSummary[M], T <: ModelTransformer[M, T]]
  (
    override val uid: String)
    extends ModelTransformer[M, T] with DefaultParamsWritable {

    override def transform(dataset: DataFrame): DataFrame = dataset

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema
  }

  /**
    * Utility simplifying transformations when data transformation is provided externally.
    *
    * @param dataTransformer Transformer for data.
    */
  abstract class PredefinedDataTransformer[M <: ModelWithSummary[M], T <: ModelTransformer[M, T]]
  (
    override val uid: String,
    dataTransformer: Transformer)
    extends ModelTransformer[M, T] with DefaultParamsWritable {

    def this(dataTransformer: Transformer, modelTransformer: T) =
      this(Identifiable.randomUID("predefinedTransfomrer"), dataTransformer)

    override def transform(dataset: DataFrame): DataFrame = dataTransformer.transform(dataset)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = dataTransformer.transformSchema(schema)
  }

  /**
    * Model transformer applying transformation only to data, keeping the model unchanged.
    *
    * @param dataTransformer Transformer for data.
    */
  class IdentityModelTransformer[M <: ModelWithSummary[M]]
  (
    override val uid: String,
    dataTransformer: Transformer) extends
    PredefinedDataTransformer[M, IdentityModelTransformer[M]](uid, dataTransformer) {

    def this(dataTransformer: Transformer) = this(Identifiable.randomUID("identityModelTransformer"), dataTransformer)

    override def transformModel(model: M, originalData: DataFrame): M = model
  }

  /**
    * Data transformer which does nothing :)
    */
  class IdentityDataTransformer(override val uid: String) extends Transformer {
    def this() = this(Identifiable.randomUID("identityDataTransformer"))

    override def transform(dataset: DataFrame): DataFrame = dataset

    override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema

  }

  /**
    * Data transformer which adds partitioning.
    */
  class PartitioningTransformer(override val uid: String) extends Transformer with PartitioningParams {
    def this() = this(Identifiable.randomUID("partitioner"))

    val numPartitions = new Param[Int](this, "numPartitions", "Number of partitions to create")

    def setNumPartitions(value: Int): this.type = set(numPartitions, value)

    override def transform(dataset: DataFrame): DataFrame = {
      val partitioned = if (isDefined(partitionBy) && !$(partitionBy).isEmpty) {
        if (isDefined(numPartitions)) {
          dataset.repartition($(numPartitions), $(partitionBy).map(dataset(_)): _*)
        } else {
          dataset.repartition($(partitionBy).map(dataset(_)): _*)
        }
      } else if (isDefined(numPartitions)) {
        dataset.repartition($(numPartitions))
      } else {
        dataset
      }

      if (isDefined(sortBy) && !$(sortBy).isEmpty) {
        partitioned.sortWithinPartitions($(sortBy).map(partitioned(_)): _*)
      } else {
        partitioned
      }
    }

    override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema

  }

  /**
    * Data transformer for projecting.
    */
  class ProjectingTransformer(override val uid: String) extends Transformer {
    def this() = this(Identifiable.randomUID("projector"))

    val columnsToKeep = new StringArrayParam(this, "columnsToKeep", "Columns to keep in the dataset. Mutually exclusive with columns to remove")
    val columnsToRemove = new StringArrayParam(this, "columnsToRemove", "Columns to remove from the dataset. Mutually exclusive with columns to keep")

    def setColumnsToRemove(columns: Seq[String]) = set(columnsToRemove, columns.toArray)

    def setColumnsToKeep(columns: Seq[String]) = set(columnsToKeep, columns.toArray)

    override def transform(dataset: DataFrame): DataFrame = {
      if (isDefined(columnsToKeep) && !$(columnsToKeep).isEmpty) {
        dataset.select($(columnsToKeep).map(dataset(_)): _*)
      } else if (isDefined(columnsToRemove) && !$(columnsToRemove).isEmpty) {
        val toFilter = $(columnsToRemove).toSet
        $(columnsToRemove).foldLeft(dataset)((data, string) => data.drop(string))
      } else {
        dataset
      }
    }

    override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema

  }

  /**
    * Utility used to inject caching.
    */
  class CachingTransformer[M <: ModelWithSummary[M]](override val uid: String) extends ModelTransformer[M, CachingTransformer[M]] {

    def storageLevel = new Param[StorageLevel](this, "storageLevel", "Storage level to use for cached data.")

    def materializeCached = new BooleanParam(
      this, "materializeCached", "Whenever to materialize cached data. If nested estimator is parallelizable it is " +
        "worth doing. Otherwise cached data might be materialized more than once.")

    def cacheRdd = new BooleanParam(
      this, "cacheRdd", "Whenever to cache RDD and re-create DataFrame. Skips columnar serialized form of DataFrame caching," +
        " thus provides faster processing with potentially large memory footprint.")

    setDefault(materializeCached -> false, storageLevel -> StorageLevel.MEMORY_ONLY, cacheRdd -> false)

    def setMaterializeCached(value: Boolean = true): this.type = set(materializeCached, value)

    def setCacheRdd(value: Boolean = true): this.type = set(cacheRdd, value)

    def setStorageLevel(value: StorageLevel): this.type = set(storageLevel, value)

    def this() = this(Identifiable.randomUID("cacher"))

    override def transformModel(model: M, originalData: DataFrame): M = model


    override def release(originalData: DataFrame, transformedData: DataFrame): Unit =
      if ($(cacheRdd)) originalData.rdd.unpersist() else transformedData.unpersist()

    override def transform(dataset: DataFrame): DataFrame = {
      val result = if ($(cacheRdd)) {
        dataset.sqlContext.createDataFrame(
          dataset.rdd.cache(),
          dataset.schema
        )
      }
      else {
        dataset.persist($(storageLevel))
      }

      if ($(materializeCached)) {
        result.count()
      }

      result
    }

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema
  }

  /**
    * Utility used to persist portion of data into temporary storage. Usefull for grounding execution plans and avoid
    * massive "skips". Unlike chekpointing is more explicit and controllable.
    */
  class PersistingTransformer[M <: ModelWithSummary[M]](override val uid: String) extends ModelTransformer[M, PersistingTransformer[M]] {

    def tempPath = new Param[String](
      this, "tempPath", "Where to store temporary data.")

    def uncacheInput = new Param[Boolean](
      this, "uncacheInput", "Whenever to uncache the input dataset.")

    def partitionByColumns = new StringArrayParam(
      this, "partitionByColumns", "Columns to partition output in a file system by (data/key=value)." )

    def setTempPath(value: String): this.type = set(tempPath, value)

    def setPartitionBy(value: Array[String]): this.type = set(partitionByColumns, value)

    def setUncacheInput(value: Boolean): this.type = set(uncacheInput, value)

    setDefault(uncacheInput -> false, partitionByColumns -> Array())

    def this() = this(Identifiable.randomUID("persister"))

    override def transformModel(model: M, originalData: DataFrame): M = model

    override def release(originalData: DataFrame, transformedData: DataFrame): Unit = {
      val path = new Path(transformedData.inputFiles.head).getParent

      if (!FileSystem.get(transformedData.sqlContext.sparkContext.hadoopConfiguration).delete(path, true)) {
        logWarning(s"Failed to remove temporary files at ${path.toString}")
      } else {
        logInfo(s"Deleted temporary files at ${path.toString}")
      }
    }


    override def transform(dataset: DataFrame): DataFrame = {
      val myPath: String = s"${$(tempPath)}/$uid/"
      // Make sure we clean up on exit
      FileSystem.get(dataset.sqlContext.sparkContext.hadoopConfiguration).deleteOnExit(new Path(myPath))

      val path = s"$myPath${Identifiable.randomUID("data")}"

      logInfo(s"Saving and re-reading data from $path")
      dataset.write.partitionBy($(partitionByColumns) : _*).parquet(path)

      if ($(uncacheInput)) {
        dataset.unpersist()
      }

      dataset.sqlContext.read.parquet(path)
    }

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema
  }

  /**
    * Collects all summary blocks and materializes them as into a single partition.
    */
  class CollectSummaryTransformer[M <: ModelWithSummary[M]](override val uid: String)
    extends ModelOnlyTransformer[M, CollectSummaryTransformer[M]](uid) with Logging {

    def this() = this(Identifiable.randomUID("summaryCollector"))

    override def transformModel(model: M, originalData: DataFrame): M = {
      model.copy(model.summary.blocks.transform((kee, data) => {
        val collected: mutable.WrappedArray[Row] = data.collect()

        logInfo(s"For model $model block $kee collected ${collected.size} records")

        data.sqlContext.createDataFrame(
          data.sqlContext.sparkContext.parallelize(collected, 1),
          data.schema)
      }))
    }
  }

  /**
    * In case if number of partitions is not known upfront, you can use dynamic partitioner to split into partitions
    * of predefined size (approximatelly).
    */
  class DynamicPartitionerTrainer[M <: ModelWithSummary[M]]
  (
    override val uid: String)
    extends Estimator[IdentityModelTransformer[M]] with DefaultParamsWritable with PartitioningParams {

    val recordsPerPartition = new Param[Long](this,
      "recordsPerPartition",
      "Approximate amount of records to store in one partition. Number of partitions is computed dynamicaly assuming even " +
        "partitioning.")

    val maxPartitions = new Param[Int](
      this, "maxPartitions", "Maximum number of partitions to assign")

    val minPartitions = new Param[Int](
      this, "minPartitions", "Minimum number of partitions to assign")

    setDefault(maxPartitions -> Int.MaxValue, minPartitions -> 1)

    def setRecordsPerPartition(value: Long): this.type = set(recordsPerPartition, value)

    def setMaxPartitions(value: Int): this.type = set(maxPartitions, value)

    def setMinPartitions(value: Int): this.type = set(minPartitions, value)

    def this() = this(Identifiable.randomUID("dynamicPartitioningEstimator"))

    override def fit(dataset: DataFrame): IdentityModelTransformer[M] = new IdentityModelTransformer[M](
      new PartitioningTransformer()
        .setPartitionBy($(partitionBy): _*)
        .setSortByColumns($(sortBy): _*)
        .setNumPartitions(
          Math.max(
            $(minPartitions),
            Math.min($(maxPartitions), dataset.count() / $(recordsPerPartition)).toInt))
    )

    override def copy(extra: ParamMap): DynamicPartitionerTrainer[M]
    = copyValues(new DynamicPartitionerTrainer[M]())

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema
  }

  class DynamicDataTransformerTrainer[M <: ModelWithSummary[M]]
  (
    override val uid: String,
    nested: Estimator[_])
    extends Estimator[IdentityModelTransformer[M]] with DefaultParamsWritable with PartitioningParams {


    def this(nested: Estimator[_]) = this(Identifiable.randomUID("dynamicPartitioningEstimator"), nested)

    override def fit(dataset: DataFrame): IdentityModelTransformer[M] = new IdentityModelTransformer[M](
      nested.fit(dataset).asInstanceOf[Transformer]
    )

    override def copy(extra: ParamMap): DynamicDataTransformerTrainer[M]
    = copyValues(new DynamicDataTransformerTrainer[M](nested.copy(extra)))

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = nested.transformSchema(schema)
  }

  /**
    * Parameters for sampling
    */
  trait SamplerParams extends HasSeed with DefaultParamsWritable {

    val percentage = new DoubleParam(this, "percentage", "Percentage of data to sample")
    val withReplacement = new BooleanParam(this, "withReplacement", "Whenever to take each sample only once.")
    val safePositive = new BooleanParam(this, "safePositive", "Safe positive and drop only negative samples")
    val labelColumn = new Param[String](this, "labelColumn", "Name column which contain labels")

    def setPercentage(value: Double): this.type = set(percentage, value)

    def setLabelColumn(value: String): this.type = set(labelColumn, value)

    def setWithReplacement(value: Boolean): this.type = set(withReplacement, value)

    def setSafePositive(value: Boolean): this.type = set(safePositive, value)
  }

  /**
    * For training a model on data set of uncertain size ads an ability to downsample it to a pre-defined
    * size (approximatelly).
    */
  class DynamicDownsamplerTrainer
  (
    override val uid: String)
    extends Estimator[SamplingTransformer] with SamplerParams {

    val expectedRecords = new Param[Long](this,
      "expectedRecords",
      "Approximate amount of records to sample from the dataset")

    def setExpectedRecords(value: Long): this.type = set(expectedRecords, value)

    def this() = this(Identifiable.randomUID("dynamicDownsamplerEstimator"))

    override def fit(dataset: DataFrame): SamplingTransformer =
      new SamplingTransformer()
        .setPercentage(
          Math.min(
            1.0,
            $(expectedRecords).toDouble / (dataset.count() + 1))
        )

    override def copy(extra: ParamMap): DynamicDownsamplerTrainer = defaultCopy(extra)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema
  }


  /**
    * Data transformer which takes sample of the data. Resulting dataframe is constructed in a way that
    * results are non-determenistic and might vary from run to run (unless the seed is specified or with
    * replacement enabled - in these cases we fallback to default data set sampling which is determenistic).
    */
  class SamplingTransformer(override val uid: String) extends Model[SamplingTransformer] with SamplerParams {
    def this() = this(Identifiable.randomUID("sampler"))

    setDefault(
      withReplacement -> false,
      safePositive -> false,
      labelColumn -> "label"

    )

    override def transform(dataset: DataFrame): DataFrame = {
      if ($(percentage) >= 1.0) {
        dataset
      } else if ($(safePositive)){
        val lColumn = $(labelColumn)
        val percent = $(percentage)

        dataset
          .withColumn("RANDOM", rand())
          .filter(s"$lColumn = 1 OR RANDOM < $percent")
          .drop("RANDOM")
      } else if (isDefined(seed)) {
        dataset.sample($(withReplacement), $(percentage), $(seed))
      } else if ($(withReplacement)) {
        dataset.sample($(withReplacement), $(percentage))
      } else {
        val localPercentage = $(percentage)
        val random = functions.udf[Boolean](() => ThreadLocalRandom.current().nextDouble() < localPercentage)
        dataset.where(random())
      }
    }

    override def copy(extra: ParamMap): SamplingTransformer = defaultCopy(extra)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema
  }


  /**
    * For training a model on data set of uncertain size ads an ability to take only the "most recent" records.
    * Estimates the size of the dataset and calculates approximate bounds for filtering.
    */
  class OrderedCutEstimator
  (
    override val uid: String)
    extends Estimator[OrderedCut] with HasGroupByColumns {

    def this() = this(Identifiable.randomUID("orderedCutEstimator"))

    final val sortByColumn = new Param[String](
      this, "sortByColumn", "Sorting criteria for the evaluation. So far only single sort column is supported.")

    def setSortByColumn(columns: String): this.type = set(sortByColumn, columns)

    val expectedRecords = new Param[Long](this,
      "expectedRecords",
      "Approximate amount of records to sample from the dataset")

    val descending = new BooleanParam(this, "descending", "Whenever to sort in a descending order.")

    setDefault(descending -> false)

    def setExpectedRecords(value: Long): this.type = set(expectedRecords, value)

    def setDescending(value: Boolean): this.type = set(descending, value)

    override def fit(dataset: DataFrame): OrderedCut = {
      val group: Column = functions.struct($(groupByColumns).map(dataset(_)): _*)
      val sort: Column = dataset($(sortByColumn))
      val expected: Long = $(expectedRecords)

      val groupColumnName: String = s"${uid}_group"
      val sortColumnName: String = s"${uid}_sort"

      val bounds = dataset
        .groupBy(group.as(groupColumnName), sort.as(sortColumnName))
        .count()
        .repartition(functions.col(groupColumnName))
        .sortWithinPartitions(functions.col(groupColumnName), if ($(descending)) functions.col(sortColumnName).desc else functions.col(sortColumnName))
        .mapPartitions(rows => {
          val map = new mutable.HashMap[Any, (Any, Long)]()

          for (row <- rows) {
            val key = row.get(0)
            val value = () => row.get(1) -> 0

            val current = map.getOrElseUpdate(key, row.get(1) -> 0)

            if (current._2 < expected) {
              map(key) = row.get(1) -> (current._2 + row.getLong(2))
            }
          }

          map.map(x => x._1 -> x._2._1).iterator
        })
        .collect()

      logInfo(s"Got bounds configuration: ${bounds.mkString("[", ",", "]")}")

      new OrderedCut()
        .setBounds(bounds)
        .setGroupByColumns($(groupByColumns): _*)
        .setSortByColumn($(sortByColumn))
        .setDescending($(descending))
        .setParent(this)
    }

    override def copy(extra: ParamMap): OrderedCutEstimator = defaultCopy(extra)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema
  }

  /**
    * Keeps data based one the some ordered constraint.
    */
  class OrderedCut
  (
    override val uid: String)
    extends Model[OrderedCut] with HasGroupByColumns {

    def this() = this(Identifiable.randomUID("orderedCut"))

    final val sortByColumn = new Param[String](
      this, "sortByColumn", "Sorting criteria for the evaluation. So far only single sort column is supported.")

    val bounds = new Param[Array[(Any, Any)]](this, "bounds", "Pre-calculated bounds for the cut")

    val descending = new BooleanParam(this, "descending", "Whenever to sort in a descending order.")

    setDefault(descending -> false)

    def setSortByColumn(columns: String): this.type = set(sortByColumn, columns)

    def setDescending(value: Boolean): this.type = set(descending, value)

    def setBounds(value: Array[(Any, Any)]): this.type = set(bounds, value)

    override def transform(dataset: DataFrame): DataFrame = {

      val group: Column = functions.struct($(groupByColumns).map(dataset(_)): _*)
      val sort: Column = dataset($(sortByColumn))

      dataset.where(
        $(bounds)
          .map(r => {
            val key = Literal.create(r._1, new StructType($(groupByColumns).map(dataset.schema(_))))
            val bound = functions.lit(r._2)
            group === key && (if ($(descending)) sort >= bound else sort <= bound)
          })
          .reduce(_ || _)
      )
    }

    override def copy(extra: ParamMap): OrderedCut = defaultCopy(extra)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema
  }


}
