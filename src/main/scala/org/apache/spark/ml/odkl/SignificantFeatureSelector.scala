package org.apache.spark.ml.odkl

import com.google.common.base.Strings
import odkl.analysis.spark.util.Logging
import org.apache.commons.lang3.StringUtils
import org.apache.commons.math3.distribution.TDistribution
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.param.{DoubleParam, IntArrayParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types.{Metadata, MetadataBuilder, StructField, StructType}
import org.apache.spark.sql._


/**
  * Created by dmitriybugaichenko on 29.11.16.
  *
  * This utility is used to perform external feature selection based on multi-fold evaluation and computing
  * weights confidence intervals based on the weights from each fold.
  */
class FoldedFeaturesStatsAggregator[SelectingModel <: ModelWithSummary[SelectingModel] with HasWeights]
(nested: SummarizableEstimator[SelectingModel],
 override val uid: String) extends Transformer with HasFeaturesSignificance  with HasWeights with HasFeaturesCol {

  def setFeaturesCol(value: String) : this.type = set(featuresCol, value)

  def this(nested: SummarizableEstimator[SelectingModel]) = this(nested, Identifiable.randomUID("foldedFeaturesStatsAggregator"))

  override def transform(dataset: Dataset[_]): DataFrame = { val model = nested.fit(dataset)
    val weightsDf = model.summary.$(weights)

    val sig = functions.udf[Double, Double, Double, Long]((avg, std, N) => {
      val tDist = new TDistribution(N - 1)
      val critVal = tDist.inverseCumulativeProbability(1.0 - (1 - 0.95) / 2)
      val confidence = critVal * std / Math.sqrt(N)

      if(confidence <= 0.0) 0.0 else Math.abs(avg / confidence)
    })

    val grouped = model match {
      case withDescriminant: HasDescriminantColumn =>
        weightsDf.groupBy(weightsDf(feature_index), weightsDf(feature_name), weightsDf(withDescriminant.getDescriminantColumn))
      case _ =>
        weightsDf.groupBy(weightsDf(feature_index), weightsDf(feature_name))
    }

    val significance = grouped.agg(
      functions.avg(weightsDf(weight)).as(average),
      functions.stddev_samp(weightsDf(weight)).as(stdDev),
      functions.count(weightsDf(weight)).as(count),
      sig(functions.avg(weightsDf(weight)),
        functions.stddev_samp(weightsDf(weight)),
        functions.count(weightsDf(weight))).as(this.significance))
      .repartition(1)

    val statRecords = toStatRecords(significance, model)

    dataset.withColumn(
      $(featuresCol),
      dataset($(featuresCol)).as($(featuresCol),
        constructMetadata(dataset.schema($(featuresCol)), statRecords.stats)))
  }

  def toStatRecords(significance : DataFrame, model: SelectingModel) : WeightsStat = {
    val indexIndex = significance.schema.fieldIndex(feature_index)
    val nameIndex = significance.schema.fieldIndex(feature_name)
    val averageIndex = significance.schema.fieldIndex(average)
    val stdDevIndex = significance.schema.fieldIndex(stdDev)
    val countIndex = significance.schema.fieldIndex(count)
    val significanceIndex = significance.schema.fieldIndex(this.significance)

    val discriminantExtractor = model match {
      case withDescriminant: HasDescriminantColumn =>
        val index = significance.schema.fieldIndex(withDescriminant.getDescriminantColumn)
        (r: Row) => r.getString(index)
      case _ => (r: Row) => ""
    }

    WeightsStat(significance.rdd.map(r => WeightsStatRecord(
      r.getInt(indexIndex),
      r.getString(nameIndex),
      discriminantExtractor(r),
      r.getDouble(averageIndex),
      r.getDouble(stdDevIndex),
      r.getLong(countIndex),
      r.getDouble(significanceIndex),
      // Do not perform selection at this stage
      isRelevant = true)
    ).collect())
  }

  def constructMetadata(field: StructField, stats: Array[WeightsStatRecord]): Metadata = {
    val builder = new MetadataBuilder()

    builder.putMetadataArray(SignificantFeatureSelector.WEIGHTS_STAT, stats.map(_.toMetadata))

    if (field.metadata != null) {
      builder.withMetadata(field.metadata)
    }

    builder.build()
  }

  override def copy(extra: ParamMap): Transformer = new FoldedFeaturesStatsAggregator[SelectingModel](nested.copy(extra))

  override def transformSchema(schema: StructType): StructType = schema
}


abstract class SignificantFeatureSelector[Filter <: GenericFeatureSelector[Filter]]
(override val uid: String) extends SummarizableEstimator[Filter] with HasWeights with HasFeaturesCol with HasFeaturesSignificance {

  def minSignificance = new DoubleParam(this, "minSignificance", "Minimum feature significance for selecting.")

  def setMinSignificance(value: Double) : this.type = set(minSignificance, value)

  def fromStatRecords(sqlContext: SQLContext, records: Array[WeightsStatRecord]): DataFrame = {

   val data = sqlContext.createDataFrame(
      sqlContext.sparkContext.parallelize(records, 1))
     .drop("isRelevant")

    if (!records.exists(x => StringUtils.isNotBlank(x.descriminant))) {
      data.drop("descriminant")
    } else {
      data
    }
  }

  /**
    * Transforms the input dataset.
    */
  override def fit(dataset: Dataset[_]): Filter = {

    val field = dataset.schema($(featuresCol))

    if(field.metadata == null || !field.metadata.contains(SignificantFeatureSelector.WEIGHTS_STAT)) {
      logError(s"Failed to get feature significance stat from field $field at $uid")
      throw new IllegalArgumentException(s"Failed to get feature significance stat from field $field at $uid")
    }

    val stats = field.metadata
      .getMetadataArray(SignificantFeatureSelector.WEIGHTS_STAT)
      .map(x => new WeightsStatRecord(x))
      .map(x => x.copy(isRelevant = x.significance >= $(minSignificance)))

    val relevant: Array[Int] = stats.view.filter(_.isRelevant).map(_.index).distinct.sorted.toArray
    val reverseMap = relevant.zipWithIndex.toMap

    require(relevant.length > 0, "No features remained after selection")

    val result = createModel()

    result.copy(
      blocks = Map(featuresSignificance -> fromStatRecords(dataset.sqlContext, stats)),
      params = ParamMap(
        result.relevantFeatures -> relevant,
        result.weightsStat -> WeightsStat(stats),
        result.originalSize -> AttributeGroup.fromStructField(dataset.schema(getFeaturesCol)).size
      ))
  }

  protected def createModel() : Filter

  override def transformSchema(schema: StructType): StructType = schema
}

class PipelinedSignificantFeatureSelector(override val uid: String)
extends SignificantFeatureSelector[PipelinedFeatureSelector] (uid) {

  def this() = this(Identifiable.randomUID("pipelinedFoldedSelector"))

  protected def createModel() : PipelinedFeatureSelector = new PipelinedFeatureSelector()

  override def copy(extra: ParamMap): PipelinedSignificantFeatureSelector = copyValues(
    new PipelinedSignificantFeatureSelector(), extra
  )
}

class LinearModelSignificantFeatureSelector[
ResultModel <: LinearModel[ResultModel]]
(
  override val uid: String
)
  extends SignificantFeatureSelector[LinearModelUnwrappedFeatureSelector[ResultModel]] (uid) {

  def this() = this(Identifiable.randomUID("linearUnwrapperFoldedSelector"))


  override def copy(extra: ParamMap): LinearModelSignificantFeatureSelector[ResultModel] = copyValues(
    new LinearModelSignificantFeatureSelector[ResultModel](), extra)

  protected def createModel() : LinearModelUnwrappedFeatureSelector[ResultModel] = new LinearModelUnwrappedFeatureSelector[ResultModel]()
}

case class WeightsStatRecord(index: Int, name: String, descriminant: String, average: Double, stdDev: Double, count: Long, significance: Double, isRelevant: Boolean) {
  def this(metadata: Metadata) = this(
    metadata.getLong("index").toInt,
    metadata.getString("name"),
    metadata.getString("descriminant"),
    metadata.getDouble("average"),
    metadata.getDouble("stdDev"),
    metadata.getLong("count"),
    metadata.getDouble("significance"),
    metadata.getBoolean("isRelevant")
  )

  def toMetadata: Metadata = {
    val builder = new MetadataBuilder

    builder
      .putLong("index", index)
      .putString("name", Strings.nullToEmpty(name))
      .putString("descriminant", Strings.nullToEmpty(descriminant))
      .putDouble("average", average)
      .putDouble("stdDev", stdDev)
      .putLong("count", count)
      .putDouble("significance", significance)
      .putBoolean("isRelevant", isRelevant)
      .build()
  }
}



case class WeightsStat(stats: Array[WeightsStatRecord])

abstract class GenericFeatureSelector[M <: ModelWithSummary[M]] extends ModelWithSummary[M] with HasFeaturesCol {

  val originalSize = new Param[Int](this, "originalSize", "Number of features in the original data.")
  val relevantFeatures = new IntArrayParam(this, "relevantFeatures", "Features with high enough significance")
  val weightsStat = new Param[WeightsStat](this, "weightsStat", "Statistics regarding model weights acquired during selection.")

  override def transform(dataset: Dataset[_]): DataFrame = {

    val relevant: Array[Int] = $(relevantFeatures)
    val reverseMap = relevant.zipWithIndex.toMap

    val reindex = functions.udf[Vector, Vector] {
      case d: DenseVector => new DenseVector(relevant.map(i => d(i)))
      case s: SparseVector => Vectors.sparse(
        relevant.length,
        s.indices.zip(s.values).map(x => reverseMap.getOrElse(x._1, -1) -> x._2).filter(_._1 >= 0))
    }

    dataset.withColumn(
      $(featuresCol),
      reindex(dataset($(featuresCol))).as(
        $(featuresCol),
        convertMetadata(relevant, dataset.schema($(featuresCol)))))
  }

  def convertMetadata(relevant: Array[Int], field: StructField): Metadata = {
    val builder = new MetadataBuilder()

    if(isDefined(weightsStat)) {
      builder.putMetadataArray(SignificantFeatureSelector.WEIGHTS_STAT, $(weightsStat).stats.map(_.toMetadata))
    }

    if (field.metadata != null) {
      builder.withMetadata(field.metadata)

      AttributeGroup.fromStructField(field).attributes
        .map(x => new AttributeGroup(field.name, relevant.zipWithIndex.map(ij => x(ij._1).withIndex(ij._2))))
        .getOrElse(new AttributeGroup(field.name, relevant.length))
        .toMetadata(builder.build())
    } else {
      builder.build()
    }
  }

  override def transformSchema(schema: StructType): StructType = StructType(schema.map(f =>
    if (f.name == $(featuresCol)) {
      StructField(f.name, f.dataType, f.nullable, convertMetadata($(relevantFeatures), f))
    } else {
      f
    }))
}

class PipelinedFeatureSelector(override val uid: String) extends
  GenericFeatureSelector[PipelinedFeatureSelector] {

  def this() = this(Identifiable.randomUID("pipelinedFeatureSelector"))

  def create() : PipelinedFeatureSelector = new PipelinedFeatureSelector()
}

class LinearModelUnwrappedFeatureSelector[M <: LinearModel[M]](override val uid: String)
  extends GenericFeatureSelector[LinearModelUnwrappedFeatureSelector[M]] with ModelTransformer[M, LinearModelUnwrappedFeatureSelector[M]] {

  def this() = this(Identifiable.randomUID("linearModelUnwrappedFeatureSelector"))

  def transformModel(model: M, originalData: DataFrame): M = {
    val size = Math.max($(originalSize), AttributeGroup.fromStructField(originalData.schema(getFeaturesCol)).size)
    SignificantFeatureSelector.transformLinearModel(size, $(relevantFeatures))(model).copy(summary.blocks)
  }

  override def copy(extra: ParamMap): LinearModelUnwrappedFeatureSelector[M] = copyValues(create(), extra)

  protected def create(): LinearModelUnwrappedFeatureSelector[M] = new LinearModelUnwrappedFeatureSelector[M]()
}

class CombinedLinearModelUnwrappedFeatureSelector[M <: LinearModel[M], C <: CombinedModel[M, C]](override val uid: String)
  extends GenericFeatureSelector[CombinedLinearModelUnwrappedFeatureSelector[M,C]]
    with ModelTransformer[C, CombinedLinearModelUnwrappedFeatureSelector[M,C]] {

  def this() = this(Identifiable.randomUID("combinedLinearModelUnwrappedFeatureSelector"))

  def transformModel(model: C, originalData: DataFrame): C = {
    val size = Math.max($(originalSize), AttributeGroup.fromStructField(originalData.schema(getFeaturesCol)).size)
    model.transformNested(SignificantFeatureSelector.transformLinearModel(size, $(relevantFeatures))).copy(summary.blocks)
  }

  override def copy(extra: ParamMap): CombinedLinearModelUnwrappedFeatureSelector[M,C] = copyValues(create(), extra)

  protected def create(): CombinedLinearModelUnwrappedFeatureSelector[M,C] = new CombinedLinearModelUnwrappedFeatureSelector[M,C]()
}

object SignificantFeatureSelector extends Serializable with Logging {
  val WEIGHTS_STAT = "features_stat"

  def tryGetInitials(field: StructField) : Option[Vector] = {
    if(field.metadata != null && field.metadata.contains(WEIGHTS_STAT)) {
      val stat = field.metadata.getMetadataArray(WEIGHTS_STAT).map(x => new WeightsStatRecord(x))

      val dense = Vectors.dense(stat.filter(_.isRelevant).sortBy(_.index).map(_.average))
      logInfo(s"Got initial weights for field ${field.name}: $dense")
      Some(dense)
    } else {
      None
    }
  }

  def transformLinearModel[M <: LinearModel[M]](originalSize: Int, relevant: Array[Int])(model: M): M = {
    val nestedSummary: ModelSummary = model.summary

    val coefficients = new SparseVector(originalSize, relevant, model.getCoefficients.toArray)

    val summary = SparkSqlUtils.reflectionLock.synchronized(
      nestedSummary.transform(model.weights -> (data => {
        val reindex = functions.udf[Int,Int](i => if(i >= 0) relevant(i) else i)

        data.withColumn(model.index, reindex(data(model.index)))
      }))
    )

    model.copy(
      summary,
      ParamMap(model.coefficients -> coefficients))
  }

  def select[SelectingModel <: ModelWithSummary[SelectingModel] with HasWeights, ResultModel <: LinearModel[ResultModel]](
              selector: SummarizableEstimator[SelectingModel],
              estimator: SummarizableEstimator[ResultModel],
              minSignificance: Double,
              featuresCol: String = "features"
            ) : SummarizableEstimator[ResultModel] = {

    val significanceSelector = new LinearModelSignificantFeatureSelector[ResultModel]()
      .setMinSignificance(minSignificance)

    significanceSelector.set(significanceSelector.featuresCol -> featuresCol)

    UnwrappedStage.wrap(
        estimator,
        UnwrappedStage.dataOnly(
          significanceSelector,
          new FoldedFeaturesStatsAggregator[SelectingModel](selector).setFeaturesCol(featuresCol)
        )
      )
  }
}