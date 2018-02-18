package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utility for vector statistics calculation.
  */

import odkl.analysis.spark.util.collection.OpenHashMap
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.param.shared.HasInputCol
import org.apache.spark.ml.param.{DoubleArrayParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, functions}
import org.apache.spark.mllib.linalg.VectorImplicits._

/**
  * Utility used to collect detailed stat for vectors grouped by a certain keys. In addition to common stuff
  * (mean, variance, min/max, norms) calculates percentiles as configured. Resulting dataframe contains only
  * columns from the key and stat columns.
  *
  * @param uid
  */
class VectorStatCollector(override val uid: String) extends
  Transformer with HasInputCol with HasGroupByColumns with DefaultParamsWritable {

  val percentiles = new DoubleArrayParam(this, "percentiles", "Percentiles to calculate for the vectors.")

  val dimensions = new Param[Int](this, "dimensions", "Dimensionality of vectors to aggregate. Taken from metadata if not provided.")

  val compression = new Param[Int](this, "compression",
    "How should accuracy be traded for size?  A value of N here will give quantile errors almost always less " +
      "than 3/N with considerably smaller errors expected for extreme quantiles.  Conversely, you should " +
      "expect to track about 5 N centroids for this accuracy.")

  val numPartitions = new Param[Int](this, "numPartitions", "Number of partitions for final result.")

  val numShufflePartitions = new Param[Int](
    this, "numShufflePartitions", "Number of partitions used for intermediate shuffle. In case if there are only a few keys" +
      " in the result this could improve performance by adding an intermediate combiner.")

  setDefault(
    groupByColumns -> Array(),
    percentiles -> Array(0.1, 0.5, 0.9),
    compression -> 150
  )

  def setInputCol(column: String): this.type = set(inputCol, column)

  def setNumPartitions(value: Int): this.type = set(numPartitions, value)

  def setNumShufflePartitions(value: Int): this.type = set(numShufflePartitions, value)

  def setCompression(value: Int): this.type = set(compression, value)

  def setDimensions(value: Int): this.type = set(dimensions, value)

  def setPercentiles(value: Array[Double]) : this.type = set(percentiles, value)

  def this() = this(Identifiable.randomUID("vectorsStatCollector"))

  override def transform(dataset: Dataset[_]): DataFrame = {

    val dimensionsVal = get(dimensions).getOrElse(AttributeGroup.fromStructField(dataset.schema($(inputCol))).size)
    val compressionVal = $(compression)
    val numPartitionsVal = get(numPartitions).getOrElse(dataset.sqlContext.sparkContext.defaultParallelism)

    val hasGrouping = isDefined(groupByColumns) && !$(groupByColumns).isEmpty

    val groupingExpression = if (hasGrouping) functions.struct($(groupByColumns).map(dataset(_)): _*)
      else functions.struct(functions.lit(1).as("lit"))


    val preAggregated: RDD[(Row, ExtendedMultivariateOnlineSummarizer)] = dataset.toDF
      .select(groupingExpression, dataset($(inputCol)))
      .rdd
      .mapPartitions(data => {
        val aggregate = new OpenHashMap[Row, ExtendedMultivariateOnlineSummarizer]()

        for (r <- data) {
          aggregate.changeValue(
            r.getAs[Row](0),
            new ExtendedMultivariateOnlineSummarizer(dimensionsVal, compressionVal).add(r.getAs[Vector](1)),
            v => v.add(r.getAs[Vector](1)))
        }

        aggregate.iterator
      })

    val mayBeShuffled: RDD[(Row, ExtendedMultivariateOnlineSummarizer)] = if(isDefined(numShufflePartitions)) {
      preAggregated.repartition($(numShufflePartitions))
    } else {
      preAggregated
    }

    dataset.sqlContext.createDataFrame(
      mayBeShuffled
        .reduceByKey((a, b) => a.merge(b), numPartitionsVal)
        .map(x => {
          val row = Row.fromSeq(Seq(
            x._2.count,
            x._2.mean.asML,
            x._2.variance.asML,
            x._2.min.asML,
            x._2.max.asML,
            x._2.numNonzeros.asML,
            x._2.normL1.asML,
            x._2.normL2.asML) ++ $(percentiles).map(p => x._2.percentile(p))
          )

          if (hasGrouping) Row.merge(x._1, row) else row}
        ),
      transformSchema(dataset.schema))

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {

    Map("" -> 0.0).map(x => x)

    logInfo(s"Input schema $schema")

    StructType(
      $(groupByColumns).map(f => schema.fields(schema.fieldIndex(f))) ++ Seq(
        StructField(s"${$(inputCol)}_count", LongType, nullable = false, metadata = schema($(inputCol)).metadata),
        StructField(s"${$(inputCol)}_mean", new VectorUDT, nullable = false, metadata = schema($(inputCol)).metadata),
        StructField(s"${$(inputCol)}_var", new VectorUDT, nullable = false, metadata = schema($(inputCol)).metadata),
        StructField(s"${$(inputCol)}_min", new VectorUDT, nullable = false, metadata = schema($(inputCol)).metadata),
        StructField(s"${$(inputCol)}_max", new VectorUDT, nullable = false, metadata = schema($(inputCol)).metadata),
        StructField(s"${$(inputCol)}_nonZeros", new VectorUDT, nullable = false, metadata = schema($(inputCol)).metadata),
        StructField(s"${$(inputCol)}_L1", new VectorUDT, nullable = false, metadata = schema($(inputCol)).metadata),
        StructField(s"${$(inputCol)}_L2", new VectorUDT, nullable = false, metadata = schema($(inputCol)).metadata)
      ) ++ $(percentiles).map(p => StructField(f"${$(inputCol)}_p${(p * 100).toInt}", new VectorUDT, nullable = false, metadata = schema($(inputCol)).metadata))
    )
  }
}
