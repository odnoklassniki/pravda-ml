package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains classes for combined models training.
  */

import java.io.IOException

import org.apache.hadoop.fs.{FileSystem, Path, PathFilter}
import org.apache.spark.annotation.Since
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.odkl.ModelWithSummary.{WithSummaryReader, WithSummaryReaderUntyped, WithSummaryWriter}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReader, Identifiable, MLReadable, MLReader}
import org.apache.spark.ml.{PipelineStage, PredictionModel, Transformer}
import org.apache.spark.ml.linalg._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types.{DoubleType, Metadata, StructField, StructType}
import org.apache.spark.util.collection.CompactBuffer

import scala.util.Try


object CombinedModel extends MLReadable[PipelineStage] {

  /**
    * Train dedicated model for a certain type. Result is a selector model. Summary blocks for combined model are
    * merged from the nested model with extra type column.
    *
    * @param estimator  Estimator for training the nested models.
    * @param typeColumn Column where the type label stored.
    * @param numThreads Number of concurent types training.
    * @param cacheForks Whenever to cache forks before iterating
    * @tparam M Type of the model to train.
    * @return Selector model (given instance type applies ONE of the models)
    */
  def perType[M <: ModelWithSummary[M]]
  (
    estimator: SummarizableEstimator[M],
    typeColumn: String = "type",
    numThreads: Int = 1,
    cacheForks: Boolean = false): PerTypeModelLearner[M] = {
    new PerTypeModelLearner[M](estimator)
      .setNumThreads(numThreads)
      .setTypeColumn(typeColumn)
      .setCacheForks(cacheForks)
  }

  /**
    * Train models for all classes. Result is a linear combination model. Summary blocks for combined model are
    * merged from the nested model with extra class column.
    *
    * @param estimator       Estimator for training the nested models.
    * @param classesColumn   Column where the classes are recorded.
    * @param classesToIgnore Which classes not to include into result.
    * @param classesMap      Mapping for merging/renaming classes.
    * @param numThreads      Number of concurrent types training.
    * @param cacheForks      Whenever to cache forks before iterating
    * @tparam M Type of the model to train.
    * @return Linear combination model (given an instance coputes prediction of all the models and combines them linary
    *         with configured weights).
    */
  def linearCombination[M <: ModelWithSummary[M]]
  (
    estimator: SummarizableEstimator[M],
    classesColumn: String = "classes",
    classesToIgnore: Seq[String] = Seq(),
    classesMap: Map[String, String] = Map(),
    numThreads: Int = 1,
    cacheForks: Boolean = false): LinearCombinationModelLearner[M] = {
    new LinearCombinationModelLearner[M](estimator)
      .setClassesToIgnore(classesToIgnore: _*)
      .setClassesMap(classesMap.toSeq: _*)
      .setClassesColumn(classesColumn)
      .setNumThreads(numThreads)
      .setCacheForks(cacheForks)
  }

  /**
    * Train models for all classes. Result is a vector with prediction for each class. Summary blocks for combined model are
    * merged from the nested model with extra class column.
    *
    * @param estimator       Estimator for training the nested models.
    * @param classesColumn   Column where the classes are recorded.
    * @param classesToIgnore Which classes not to include into result.
    * @param classesMap      Mapping for merging/renaming classes.
    * @param numThreads      Number of concurrent class training.
    * @param cacheForks      Whenever to cache forks before iterating
    * @tparam M Type of the model to train.
    * @return Linear combination model (given an instance coputes prediction of all the models and combines them linary
    *         with configured weights).
    */
  def multiClass[M <: ModelWithSummary[M]]
  (
    estimator: SummarizableEstimator[M],
    classesColumn: String = "classes",
    classesToIgnore: Seq[String] = Seq(),
    classesMap: Map[String, String] = Map(),
    numThreads: Int = 1,
    cacheForks: Boolean = false): MultiClassModelLearner[M] = {
    new MultiClassModelLearner[M](estimator)
      .setClassesToIgnore(classesToIgnore: _*)
      .setClassesMap(classesMap.toSeq: _*)
      .setClassesColumn(classesColumn)
      .setNumThreads(numThreads)
      .setCacheForks(cacheForks)
  }


  /**
    * @return Reader for type selecting model.
    */
  def perTypeReader[M <: ModelWithSummary[M]](nestedReader: WithSummaryReader[M]): WithSummaryReader[SelectingModel[M]] = {
    new CombinedModel.Reader[M, SelectingModel[M]]()
  }

  /**
    * @return Reader for linear combination combination model.
    */
  def linearCombinationReader[M <: ModelWithSummary[M]](nestedReader: WithSummaryReader[M]): WithSummaryReader[LinearCombinationModel[M]] = {
    new CombinedModel.Reader[M, LinearCombinationModel[M]]()
  }

  /**
    * @return Reader for multi class  combination model.
    */
  def multiClassReader[M <: ModelWithSummary[M]](nestedReader: WithSummaryReader[M]): WithSummaryReader[LinearCombinationModel[M]] = {
    new CombinedModel.Reader[M, LinearCombinationModel[M]]()
  }

  class Writer[M <: ModelWithSummary[M], C <: CombinedModel[M, C]](instance: CombinedModel[M, C])
    extends WithSummaryWriter[C](instance) {
    @throws[IOException]("If the input path already exists but overwrite is not enabled.")
    @Since("1.6.0")
    override def save(path: String): Unit = {
      super.save(path)
      // TODO: Think how to save models in-place in params if they have only own parameters.
      instance.nested.foreach(x => x._2.disableSaveSummary().write.save(s"$path/model=${x._1}"))
    }
  }

  class Reader[M <: ModelWithSummary[M], C <: CombinedModel[M, C]]
    extends WithSummaryReader[C] {
    override def load(path: String): C = {
      new ReaderUntyped().load(path).asInstanceOf[C]
    }
  }

  class ReaderUntyped extends WithSummaryReaderUntyped {

    override def load(path: String): PipelineStage = {

      val fs = FileSystem.get(sqlContext.sparkContext.hadoopConfiguration)
      val files = fs.listStatus(new Path(path), new PathFilter {
        override def accept(path: Path): Boolean = path.getName.startsWith("model=")
      })


      super.load(path) match {
        case original: CombinedModel[_, _] =>
          val nested = files.par
            .map(x => x.getPath.getName.split("=", 2)(1) -> DefaultParamsReader.loadParamsInstance[PipelineStage](x.getPath.toString, sc))
            .toMap

          original.nested = nested.seq

          if (original.isSaveSummaryEnabled) {
            original.propagateBlocks()
          }

          original
      }
    }
  }

  @Since("1.6.0")
  override def read: MLReader[PipelineStage] = new ReaderUntyped

  /**
    * Collects types, available in the dataset and trains one model per each type.
    *
    * @param nested Nested estimator or fitting.
    */
  class PerTypeModelLearner[M <: ModelWithSummary[M]]
  (
    nested: SummarizableEstimator[M],
    override val uid: String
  )
    extends ForkedEstimator[M, String, SelectingModel[M]](nested, uid) with HasTypeCol with HasPredictionCol {

    def setPredictionCol(value: String) : this.type = set(predictionCol, value)

    def this(nested: SummarizableEstimator[M]) = this(nested, Identifiable.randomUID("typeSelector"))

    override def copy(extra: ParamMap) = copyValues(new PerTypeModelLearner[M](nested.copy(extra)), extra)

    protected override def createForks(dataset: Dataset[_]): Seq[(String, DataFrame)] = {
      val types = dataset.select($(typeColumn)).distinct().collect().map(_.get(0).toString).toSeq.sorted

      logInfo(s"Got types: $types")

      types.map(t => t -> dataset.toDF.filter(dataset($(typeColumn)) === t))
    }

    protected def mergeModels(sqlContext: SQLContext, models: Seq[(String, Try[M])]): SelectingModel[M] = {
      val result = new SelectingModel[M](models.map(x => x._1 -> x._2.get).toMap, $(typeColumn))
      result.set(result.predictionCol, $(predictionCol))
    }
  }

  /**
    * Collects all classes in the dataset and trains a model for each class. Base class for multi-class
    * and linear combination model.
    *
    * @param nested Estimator for nested models.
    */
  abstract class PerClassModelLearner[N <: ModelWithSummary[N], M <: MultiClassCombinationModelBase[N,M]]
  (
    nested: SummarizableEstimator[N],
    override val uid: String
  )
    extends ForkedEstimator[N, String, M](nested, uid)
      with HasClassesCol with HasClassesWeights with HasLabelCol with HasPredictionCol {

    final val classesToIgnore: JacksonParam[Set[String]] = JacksonParam[Set[String]](
      this, "classesToIgnore", "Classes we do not want to train model for.")


    final val classesMap: JacksonParam[Map[String, String]] = JacksonParam.mapParam[String](
      this, "classesMap", "Map used to rename classes (e.g. merging Complain and Dislike).")

    setDefault(classesToIgnore, Set[String]())
    setDefault(classesMap, Map[String, String]())

    def setClassesToIgnore(values: String*): this.type = set(classesToIgnore, values.toSet)

    def setClassesMap(values: (String, String)*): this.type = set(classesMap, values.toMap)

    def setPredictionCol(value: String) : this.type = set(predictionCol, value)

    protected override def createForks(dataset: Dataset[_]): Seq[(String, DataFrame)] = {
      val classesField = dataset.schema($(classesColumn))

      if (classesField.dataType.isInstanceOf[VectorUDT]) {
        val attributes: Array[Attribute] = AttributeGroup
          .fromStructField(classesField)
          .attributes
          .getOrElse(throw new IllegalArgumentException("Metadata required to extract class names"))

        val classesIndex = attributes.map(x => x.name.getOrElse(x.index.get.toString) -> x.index.get).toMap
        val classes = attributes.map(x => x.name.getOrElse(x.index.get.toString))

        require(!isDefined(classesMap) || $(classesMap).isEmpty, "Can not apply classes map when working with a vector.")

        val result = classes.filterNot($(classesToIgnore)).map(t => {
          val index = classesIndex(t)

          val label = SparkSqlUtils.reflectionLock.synchronized(
            functions.udf[Double, Vector](x => x(index)))

          t -> dataset.withColumn($(labelCol), label(dataset($(classesColumn))))
        })

        result
      } else {
        val classes = dataset.select($(classesColumn)).rdd.flatMap {
          case Row(multiple: Seq[Any]) => multiple.map(_.toString)
          case Row(single: Any) => Seq(single.toString)
        }.distinct().collect().toSeq.sorted

        logInfo(s"Got classes: $classes")

        val mapping = $(classesMap)
        val activeClasses: Seq[String] = classes.filterNot($(classesToIgnore)).map(x => mapping.getOrElse(x, x)).distinct.sorted

        val result = activeClasses.map(t => {

          val label = SparkSqlUtils.reflectionLock.synchronized(
            functions.udf[Double, Any]({
              case multiple: Seq[Any] => if (multiple.exists(x => mapping.getOrElse(x.toString, x.toString).equals(t))) 1.0 else 0.0
              case single: Any => if (t.equals(mapping.getOrElse(single.toString, single.toString))) 1.0 else 0.0
              case _ => 0.0
            }))

          t -> dataset.withColumn($(labelCol), label(dataset($(classesColumn))))
        })

        result
      }
    }
  }

  /**
    * Trains a model which predicts vector (one element per class)
    */
  class MultiClassModelLearner[M <: ModelWithSummary[M]]
  (
    nested: SummarizableEstimator[M],
    override val uid: String
  )
    extends PerClassModelLearner[M, MultiClassCombinationModel[M]](nested, uid) {

    def this(nested: SummarizableEstimator[M]) = this(nested, Identifiable.randomUID("multiClassLearner"))

    override def copy(extra: ParamMap): MultiClassModelLearner[M] = copyValues(new MultiClassModelLearner[M](nested.copy(extra)), extra)

    protected def mergeModels(sqlContext: SQLContext, models: Seq[(String, Try[M])]): MultiClassCombinationModel[M] = {
      val result: MultiClassCombinationModel[M] =
        new MultiClassCombinationModel[M](models.map(x => x._1 -> x._2.get).toMap)

      result
        .set(result.classes, models.map(_._1).toArray)
        .set(result.predictionCol, $(predictionCol))
    }
  }

  /**
    * Trains a model which predicts a scalar of weighted classes or a vector which is a generalized linear combination
    * of per-class predictions
    */
  class LinearCombinationModelLearner[M <: ModelWithSummary[M]] (
                                                                  nested: SummarizableEstimator[M],
                                                                  override val uid: String
                                                                )
    extends PerClassModelLearner[M, LinearCombinationModel[M]](nested, uid) {

    def this(nested: SummarizableEstimator[M]) = this(nested, Identifiable.randomUID("linearCombinationLearner"))

    override def copy(extra: ParamMap): SummarizableEstimator[LinearCombinationModel[M]] =
      copyValues(new LinearCombinationModelLearner[M](nested.copy(extra)), extra)

    override protected def mergeModels(sqlContext: SQLContext, models: Seq[(String, Try[M])]): LinearCombinationModel[M] = {
      val result: LinearCombinationModel[M] =
        new LinearCombinationModel[M](models.map(x => x._1 -> x._2.get).toMap)

      result
        .set(result.classes, models.map(_._1).toArray)
        .set(result.predictionCol, $(predictionCol))
        .set(result.predictCombinations, Array($(classesWeights)))
    }
  }

  /**
    * Direct prediction model is an extension to a model with an ability to predict for single point instead
    * of transforming dataset. Used to improve performance of the combined models.
    *
    * @tparam I Type of the model input
    */
  trait DirectPredictionModel[I, M <: DirectPredictionModel[I, M]] extends PredictionModel[I, M] with HasDirectTransformOption {

    def predictDirect(features: Any): Double = predict(features.asInstanceOf[I])

    override protected def predict(features: I): Double

    /**
      * @return If possible, try to evaluate prediction directly, overwise revert to chain of transformations.
      */
    def directTransform(data: DataFrame): Option[Column] = SparkSqlUtils.reflectionLock.synchronized {
      val udf = functions.udf((x: Any) => predictDirect(x))

      Some(udf(data($(featuresCol))))
    }
  }

}

/**
  * Supplementary train used for optimization (moving transformation out of the execution plan into UDF)
  */
trait HasDirectTransformOption extends Transformer {

  /**
    * @return If possible, try to evaluate prediction directly, overwise revert to chain of transformations.
    */
  def directTransform(data: DataFrame): Option[Column]
}

/**
  * Base class for combined model holding a named map of nested models.
  */
abstract class CombinedModel[M <: ModelWithSummary[M], C <: CombinedModel[M, C]](nestedModels: Map[String, M])
  extends ModelWithSummary[C] with HasDescriminantColumn with HasDirectTransformOption
    with HasPredictionCol with ForkedModelParams {

  private var nestedMap: Map[String, M] = nestedModels

  def nested = nestedMap

  def setPredictionCol(value: String) : this.type = set(predictionCol, value)

  private def nested_=(nestedModels: Map[String, PipelineStage]): Unit = nestedMap = nestedModels.transform((name, model) => model.asInstanceOf[M])

  private def propagateBlocks(): Unit = {
    nested.foreach(m => {
      val blocks = m._2.summary.blocks.filter(x => this.summary.blocks(x._1).schema.fieldNames.indexOf($(descriminantColumn)) >= 0).keys.map(k => {
        val data = this.summary.blocks(k)

        k -> data.filter(data($(descriminantColumn)) === functions.lit(m._1)).drop($(descriminantColumn))
      }).toMap

      m._2.setSummary(m._2.summary.copy(blocks))
    })
  }

  override private[odkl] def setSummary(summary: ModelSummary): C = {
    val result = super.setSummary(summary)
    propagateBlocks()
    result
  }

  protected def escalateBlocks(): Unit = {
    val extendedBlocks = nested.map(x => x._2.summary.blocks.mapValues(_.withColumn($(descriminantColumn), functions.lit(x._1))))
      .reduce((a, b) => {
        a ++ b.map(x => (x._1, if (a.contains(x._1)) x._2.unionAll(a(x._1)) else x._2))
      })


    set(summaryParam, new ModelSummary(summary.blocks ++ extendedBlocks))
  }

  private[odkl] def transformNested(transformer: M => M): this.type = {
    nestedMap = nestedMap.transform((key, model) => transformer(model).asInstanceOf[M])
    escalateBlocks()
    this
  }


  /**
    * In case if direct transformation is not possible, applies less efficientt indirect logic.
    */
  protected def indirectTransform(dataset: DataFrame): DataFrame

  override def transform(dataset: Dataset[_]): DataFrame = {
    directTransform(dataset.toDF).map {
      func: Column =>
        dataset.withColumn($(predictionCol), func.as($(predictionCol), createPredictionMetadata()))
    }.getOrElse(indirectTransform(dataset.toDF))
  }

  override protected def create(): C = throw new NotImplementedError()

  @Since("1.6.0")
  override def write: WithSummaryWriter[C] = new CombinedModel.Writer[M, C](this)

  def createPredictionMetadata(): Metadata
}

/**
  * Selecting model applies exactly one model based on instance type and return its result.
  *
  * @param nestedModels Map with nested models to apply.
  */
class SelectingModel[N <: ModelWithSummary[N]]
(
  nestedModels: Map[String, N],
  override val uid: String,
  val unused: String
) extends CombinedModel[N, SelectingModel[N]](nestedModels) with HasTypeCol with HasFeaturesCol {

  def this(uid: String) = this(Map(), uid, "")

  def this(nested: Map[String, N], typeColumn: String) = {
    this(nested, Identifiable.randomUID("typeSelectingModel"), "")
    setTypeColumn(typeColumn)
    escalateBlocks()
  }

  override def descriminantColumn: Param[String] = typeColumn

  override def copy(extra: ParamMap): SelectingModel[N] = {
    val copy = new SelectingModel[N](nested.transform((k, x) => x.copy(extra)), extra.getOrElse(typeColumn, $(typeColumn)))
    copyValues(copy, extra)
  }

  /**
    * @return If possible, try to evaluate prediction directly, overwise revert to chain of transformations.
    */
  override def directTransform(data: DataFrame): Option[Column] = SparkSqlUtils.reflectionLock.synchronized {

    val `type` = data($(typeColumn))

    nested.values.head match {
      case combination: HasDirectTransformOption if combination.directTransform(data).isDefined =>
        val features = data($(featuresCol))
        val apply: Column = combination.directTransform(data).get

        Some(nested.drop(1).foldLeft
        (functions.when(`type` === nested.head._1, apply))((column, f) =>
          column.when(`type` === f._1, f._2.asInstanceOf[HasDirectTransformOption].directTransform(data).get))
          .otherwise(functions.lit(null)))

      case _ => None
    }
  }

  def createPredictionMetadata(): Metadata = {
    nested.values.head match {
      case combination: CombinedModel[_, _] => combination.createPredictionMetadata()
      case _ => Metadata.empty
    }
  }

  override protected def indirectTransform(dataset: DataFrame): DataFrame = {
    val typeCol = dataset($(typeColumn))
    nested
      .map(x => x._2.transform(mayBePropagateKey(dataset.filter(typeCol === x._1), x._1)))
      .reduce((a, b) => a.unionAll(b))
  }

  override def transformSchema(schema: StructType): StructType = {
    nested.values.head.transformSchema(schema)
  }
}

/**
  * Base class for models, evaluated per each class.
  */
abstract class MultiClassCombinationModelBase[N <: ModelWithSummary[N], M <: MultiClassCombinationModelBase[N,M]]
(
  nestedModels: Map[String, N],
  override val uid: String
) extends CombinedModel[N, M](nestedModels)
  with HasClassesCol with HasPredictionCol with HasFeaturesCol {

  final val classes = new StringArrayParam(this, "classes", "Sequence of the classes to predict. Used to enforce certain order of fields in the result.")

  def this(uid: String) = this(Map[String, N](), uid)

  override def descriminantColumn: Param[String] = classesColumn

  /**
    * @return If possible, try to evaluate prediction directly, overwise revert to chain of transformations.
    */
  def directTransform(dataFrame: DataFrame): Option[Column] = SparkSqlUtils.reflectionLock.synchronized {
    nested.values.head match {
      case firstLinear: LinearModel[_] =>
        val indexedModels = $(classes).map(x => nested(x).asInstanceOf[LinearModel[_]])

        val allCoefficients = indexedModels.map(_.getCoefficients.toArray)

        val matrix = Matrices.dense(
          allCoefficients.head.length,
          allCoefficients.length,
          allCoefficients.reduce((a, b) => a ++ b)).transpose

        val intercept = Vectors.dense(indexedModels.map(_.getIntercept))

        val column = functions.udf[Vector, Vector]((x: Vector) => {
          val y = intercept.copy.toDense
          BLAS.gemv(1.0, matrix, x, 1.0, y)

          Vectors.dense(y.toArray.transform(y => firstLinear.postProcess(y)).toArray)
        }).apply(dataFrame($(featuresCol)))

        Some(column)
      case firstDirect: HasDirectTransformOption if firstDirect.directTransform(dataFrame).isDefined =>
        val toVector = functions.udf((x: Seq[Double]) => Vectors.dense(x.toArray))
        val columns = nested.map(_.asInstanceOf[HasDirectTransformOption].directTransform(dataFrame).get)
        val array = functions.array(columns.toSeq: _*)

        Some(toVector(array))
      case _ => None
    }
  }

  override def createPredictionMetadata(): Metadata = {
    new AttributeGroup(
      $(predictionCol),
      $(classes)
        .map(x => NumericAttribute.defaultAttr.withName(x).asInstanceOf[Attribute])).toMetadata()
  }

  override def transformSchema(schema: StructType): StructType = {
    schema.add(StructField($(predictionCol), new VectorUDT(), nullable = false, createPredictionMetadata()))
  }

  override protected def indirectTransform(dataset: DataFrame): DataFrame = {
    val indexed: Array[(String, N)] = $(classes).map(x => x -> nested(x))

    var names = new CompactBuffer[String]()

    val originalColumns = dataset.columns.toSet

    val global = indexed.foldLeft(dataset)((data, model) => {

      val transformed: DataFrame = model._2.transform(mayBePropagateKey(data, model._1))

      val alias = $(predictionCol) + s"_${model._1}"
      val selected = transformed
        .select(transformed.columns.filter(originalColumns).map(x => transformed(x))
          ++ names.map(x => transformed(x))
          ++ Seq(transformed($(predictionCol)).as(alias, transformed.schema($(predictionCol)).metadata)): _*)

      names += alias

      selected
    })

    val vectorized = new AutoAssembler()
      .setColumnsToInclude(names: _*)
      .setColumnAttributeMap(indexed.map(x => $(predictionCol) + s"_${x._1}" -> x._1): _*)
      .setOutputCol($(predictionCol))
      .fit(global).transform(global)

    vectorized
      .select(dataset.columns.map(x => vectorized(x)) ++ Seq(vectorized($(predictionCol))): _*)
  }
}

/**
  * Combination model which evaluates ALL nested model and combines results based on linear weights.
  */
class LinearCombinationModel[N <: ModelWithSummary[N]]
(
  nestedModels: Map[String, N],
  override val uid: String
) extends MultiClassCombinationModelBase[N, LinearCombinationModel[N]](nestedModels, uid) {

  val predictCombinations: JacksonParam[Array[Map[String, Double]]] = JacksonParam.arrayParam[Map[String, Double]](
    this, "predictCombinations", "Whether to predict values for multiple combinations. Compatible only" +
      " with linear nested models.")

  setDefault(predictCombinations -> Array(Map[String,Double]().withDefaultValue(1.0)))

  val classesWeights = JacksonParam.mapParam[Double](this, "classesWeights", "")

  def setPredictVector(value: Map[String, Double]*): this.type = set(predictCombinations, value.toArray)

  def this(uid: String) = this(Map[String, N](), uid)

  def this(nested: Map[String, N]) = {
    this(nested, Identifiable.randomUID("linearCombinationModel"))
    escalateBlocks()
  }

  override def copy(extra: ParamMap): LinearCombinationModel[N] = {
    val copy = new LinearCombinationModel[N](nested.transform((k, x) => x.copy(extra).setParent(x.parent)))
    copyValues(copy, extra)
  }

  /**
    * @return If possible, try to evaluate prediction directly, overwise revert to chain of transformations.
    */
  override def directTransform(data: DataFrame): Option[Column] = SparkSqlUtils.reflectionLock.synchronized {
    nested.values.head match {
      case firstLinear: LinearModel[_] =>
        val indexedModels = nested.iterator.map {
          case (name: String, model: LinearModel[_]) => name -> model
        }.toArray

        val allCoefficients = indexedModels.map(_._2.getCoefficients.toArray)

        val matrix = Matrices.dense(
          allCoefficients.head.length,
          allCoefficients.length,
          allCoefficients.reduce((a, b) => a ++ b)).transpose

        val intercept = Vectors.dense(indexedModels.map(_._2.getIntercept))

        val combineUdf = if ($(predictCombinations).length == 1) {
          val weights = Vectors.dense(indexedModels.map(x => $(predictCombinations)(0)(x._1)))

          functions.udf[Double, Vector]((x: Vector) => {
            val y = intercept.copy.toDense
            BLAS.gemv(1.0, matrix, x, 1.0, y)

            y.toArray.transform(y => firstLinear.postProcess(y))

            BLAS.dot(y, weights)
          })
        } else {
          val combinations = $(predictCombinations)

          val combinationsMatrix = Matrices.dense(
            combinations.length,
            nested.size,
            nested.iterator.flatMap(x => combinations.map(_.getOrElse(x._1, 0.0))).toArray)


          functions.udf[Vector, Vector]((x: Vector) => {
            val y = intercept.copy.toDense
            BLAS.gemv(1.0, matrix, x, 1.0, y)

            y.toArray.transform(firstLinear.postProcess)

            val result = Vectors.zeros(combinations.length).toDense

            BLAS.gemv(1.0, combinationsMatrix, y, 1.0, result)

            result
          })
        }

        Some(combineUdf(data($(featuresCol))))

      case _ => super.directTransform(data).map(vector => {
        val combineUdf: UserDefinedFunction = getCombineUdf

        combineUdf(vector)
      })
    }
  }


  private def getCombineUdf = {
    val combinations = $(predictCombinations)

    val combineUdf = if (combinations.length == 1) {
      val combinationsVector = Vectors.dense(
        nested.iterator.map(x => combinations(0).getOrElse(x._1, 0.0)).toArray)

      functions.udf((x: Vector) => BLAS.dot(combinationsVector, x))
    } else {
      val combinationsMatrix = Matrices.dense(
        combinations.length,
        nested.size,
        nested.iterator.flatMap(x => combinations.map(_.getOrElse(x._1, 0.0))).toArray)

      functions.udf((x: Vector) => {
        val result = Vectors.zeros(combinations.length).toDense
        BLAS.gemv(1.0, combinationsMatrix, x, 1.0, result)
        result
      })
    }
    combineUdf
  }

  override protected def indirectTransform(dataset: DataFrame): DataFrame = {
    val withVector = super.indirectTransform(dataset)

    withVector.withColumn(
      $(predictionCol),
      getCombineUdf(withVector($(predictionCol))).as($(predictionCol), withVector.schema($(predictionCol)).metadata))
  }

  override def createPredictionMetadata(): Metadata = {
    new AttributeGroup(
      $(predictionCol),
      $(predictCombinations)
        .map(_.toString())
        .map(x => NumericAttribute.defaultAttr.withName(x).asInstanceOf[Attribute])).toMetadata()
  }


  override def transformSchema(schema: StructType): StructType = {
    if ($(predictCombinations).length == 1) {
      schema.add(StructField($(predictionCol), DoubleType, nullable = false, createPredictionMetadata()))
    } else {
      schema.add(StructField($(predictionCol), new VectorUDT(), nullable = false, createPredictionMetadata()))
    }
  }
}

/**
  * Combination model which evaluates ALL nested model and returns vector.
  */
class MultiClassCombinationModel[N <: ModelWithSummary[N]]
(
  nestedModels: Map[String, N],
  override val uid: String
) extends MultiClassCombinationModelBase[N, MultiClassCombinationModel[N]](nestedModels, uid) {

  def this(uid: String) = this(Map[String, N](), uid)

  override def copy(extra: ParamMap): MultiClassCombinationModel[N] = {
    val copy = new MultiClassCombinationModel[N](nested.transform((k, x) => x.copy(extra).setParent(x.parent)))
    copyValues(copy, extra)
  }

  def this(nested: Map[String, N]) = {
    this(nested, Identifiable.randomUID("multiClassCombinationModel"))
    escalateBlocks()
  }
}


