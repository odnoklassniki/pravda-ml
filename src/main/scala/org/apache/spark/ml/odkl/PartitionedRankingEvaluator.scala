package org.apache.spark.ml.odkl


import java.util
import java.util.Comparator

import breeze.numerics.log2
import org.apache.commons.lang.math.NumberUtils
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.odkl.PartitionedRankingEvaluator.Metric
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._
import org.apache.spark.util.collection.CompactBuffer

import scala.collection.immutable

/**
  * Evaluator used to compute metrics for predictions grouped by a certain criteria (typically by
  * a user id). Materializes all the predictions for a criteria in memory and calculates multiple metrics.
  * Can be used only for fine-grained grouping criteria. Supports mutli-label and multi-score cross evaluation
  * (computes metrics for each label-score combinations if provided with vectors instead of scalars).
  */
class PartitionedRankingEvaluator(override val uid: String) extends Evaluator[PartitionedRankingEvaluator](uid) with HasOutputCol with HasGroupByColumns {

  def this() = this(Identifiable.randomUID("partitionedRankingEvaluator"))

  val modelThreshold = new DoubleParam(
    this, "modelThreshold", "Threshold for model score to consider item included or not.")

  val labelThreshold = new DoubleParam(
    this, "labelThreshold", "Threshold for labels to consider item relevant or not.")

  val metrics = new Param[Seq[PartitionedRankingEvaluator.Metric]](
    this, "metrics", "Metrics to evaluate.")

  val extraColumns = new StringArrayParam(
    this, "extraColumns", "Extra columns to add to row for metrics. Comes right after score and label")

  val labelIndexColumnParam = new Param[String](
    this, "labelIndexColumnParam", "For multilabel validation name of the column to store name of label metrics computed for.")

  val scoreIndexColumnParam = new Param[String](
    this, "scoreIndexColumnParam", "For multiscore validation name of the column to store name of score metrics computed for.")

  setDefault(
    modelThreshold -> 0.5,
    labelThreshold -> 0.5,
    outputCol -> "metrics",
    labelIndexColumnParam -> "label",
    scoreIndexColumnParam -> "score",
    metrics -> Seq(),
    extraColumns -> Array())

  override def copy(extra: ParamMap): PartitionedRankingEvaluator = defaultCopy(extra)

  override def transform(dataset: Dataset[_]): DataFrame = {

    val indexedMetrics: immutable.IndexedSeq[PartitionedRankingEvaluator.Metric] = $(metrics).toIndexedSeq

    val struct: Column = functions.struct(
      Seq(dataset($(predictionCol)),
        dataset($(labelCol))) ++
        $(extraColumns).map(c => dataset(c)): _*
    )

    val attributes = constructMetadata(indexedMetrics)

    val groupedData: RelationalGroupedDataset = dataset.groupBy($(groupByColumns).map(dataset(_)): _*)


    val prediction: StructField = dataset.schema($(predictionCol))
    val label: StructField = dataset.schema($(labelCol))

    val nestedStruct: StructType = StructType(Array(
      prediction,
      label)
      ++
      $(extraColumns).map(c => dataset.schema(c)))

    val evaluator = new PartitionedRankingEvaluator.MultiLabelMetricEvaluator(
      nestedStruct,
      indexedMetrics,
      labelThreshold = $(labelThreshold),
      modelThreshold = $(modelThreshold))

    var result = groupedData.agg(evaluator(struct).as($(outputCol)))

    result = result.withColumn($(outputCol), functions.explode(result($(outputCol))))

    if (prediction.dataType.isInstanceOf[VectorUDT]) {
      result = result.withColumn(
        $(scoreIndexColumnParam),
        functions.expr(s"${$(outputCol)}.scoreIndex").as($(scoreIndexColumnParam), prediction.metadata))
    }

    if (label.dataType.isInstanceOf[VectorUDT]) {
      result = result.withColumn(
        $(labelIndexColumnParam),
        functions.expr(s"${$(outputCol)}.labelIndex").as($(labelIndexColumnParam), label.metadata))
    }

    result.withColumn($(outputCol), functions.expr(s"${$(outputCol)}.metrics").as($(outputCol), attributes))
  }

  def constructMetadata(indexedMetrics: immutable.IndexedSeq[Metric]): Metadata = {
    new AttributeGroup(
      "metrics",
      indexedMetrics.map(m => NumericAttribute.defaultAttr.withName(m.name)).toArray[Attribute]
    ).toMetadata()
  }

  def setMetrics(metric: PartitionedRankingEvaluator.Metric*): this.type = set(metrics, metric)

  def setModelThreshold(value: Double): this.type = set(modelThreshold, value)

  def setLabelThreshold(value: Double): this.type = set(labelThreshold, value)

  def setExtraColumns(column: String*): this.type = set(extraColumns, column.toArray)

  def setOutputCol(column: String): this.type = set(outputCol, column)

  def setLabelNameCol(column: String): this.type = set(labelIndexColumnParam, column)

  override def transformSchema(schema: StructType): StructType = {
    logInfo(s"Input schema $schema")

    StructType(
      $(groupByColumns).map(f => schema.fields(schema.fieldIndex(f)))
        ++ Seq(StructField($(outputCol), new VectorUDT, nullable = false))
        ++ (
        if (schema($(predictionCol)).dataType.isInstanceOf[VectorUDT])
          Seq(StructField($(scoreIndexColumnParam), IntegerType, nullable = false, metadata = schema($(predictionCol)).metadata))
        else Seq()
        )
        ++ (
        if (schema($(labelCol)).dataType.isInstanceOf[VectorUDT])
          Seq(StructField($(labelIndexColumnParam), IntegerType, nullable = false, metadata = schema($(labelCol)).metadata))
        else Seq())
    )
  }
}

object PartitionedRankingEvaluator extends Serializable {

  @transient lazy val byScoreComparator = new Comparator[ScoreLabel]() {
    override def compare(o1: ScoreLabel, o2: ScoreLabel): Int = NumberUtils.compare(o2.score, o1.score)
  }

  @transient lazy val byLabelComparator = new Comparator[ScoreLabel]() {
    override def compare(o1: ScoreLabel, o2: ScoreLabel): Int = NumberUtils.compare(o2.label, o1.label)
  }

  class CollectStruct(input: StructType) extends UserDefinedAggregateFunction {
    override def inputSchema: StructType = StructType(Array(StructField("input", input)))

    override def update(buffer: MutableAggregationBuffer, input: Row): Unit = if (input != null) {
      buffer.update(0, buffer.getAs[CompactBuffer[Row]](0) :+ input.getAs[Row](0))
    }

    override def bufferSchema: StructType = StructType(Seq(StructField("data", arrayOfInput)))

    override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
      buffer1.update(0, buffer1.getAs[CompactBuffer[Row]](0) ++ buffer2.getAs[CompactBuffer[Row]](0))
    }

    override def initialize(buffer: MutableAggregationBuffer): Unit = buffer.update(0, CompactBuffer())

    override def deterministic: Boolean = false

    override def evaluate(buffer: Row): Any = buffer.getAs[CompactBuffer[Row]](0)

    override def dataType: DataType = arrayOfInput

    def arrayOfInput: ArrayType = ArrayType(input, containsNull = false)
  }


  class MultiLabelMetricEvaluator
  (
    input: StructType,
    metrics: IndexedSeq[Metric],
    labelThreshold: Double,
    modelThreshold: Double) extends CollectStruct(input) {

    val isScoreVector = input.fields(0).dataType.isInstanceOf[VectorUDT]
    val isLabelVector = input.fields(1).dataType.isInstanceOf[VectorUDT]

    override def deterministic: Boolean = true


    def iterateLabels
    (prefix: Seq[Int],
     byScore: Iterable[ScoreLabel],
     byLabel: Array[ScoreLabel]): Seq[Seq[Any]] = {

      if (isLabelVector) {
        Array.tabulate(byScore.head.row.getAs[Vector](1).size)(i => {
          byLabel.transform(x => x.setLabel(x.row.getAs[Vector](1)(i)))
          util.Arrays.sort(byLabel, byLabelComparator)

          prefix ++ Seq(i, Vectors.dense(Array.tabulate(metrics.size) { i => metrics(i).apply(byLabel, byScore.iterator.takeWhile(_.score >= modelThreshold), byScore, labelThreshold) }))
        })
      } else {
        Seq({
          byLabel.transform(x => x.setLabel(x.row.getDouble(1)))
          util.Arrays.sort(byLabel, byLabelComparator)

          prefix ++ Seq(Vectors.dense(Array.tabulate(metrics.size) { i => metrics(i).apply(byLabel, byScore.iterator.takeWhile(_.score >= modelThreshold), byScore, labelThreshold) }))
        })
      }
    }

    override def evaluate(buffer: Row): Any = {
      val byScore = super.evaluate(buffer).asInstanceOf[Seq[Row]]
        .iterator.map(r => ScoreLabel(score = 0, label = 0, row = r)).toArray
      val byLabel = byScore.clone()

      (if (isScoreVector) {
        Array.tabulate(byScore.head.row.getAs[Vector](0).size)(i => {
          byScore.transform(x => x.setScore(x.row.getAs[Vector](0)(i)))
          util.Arrays.sort(byScore, byScoreComparator)

          iterateLabels(Seq(i), byScore, byLabel)
        }).toSeq.flatMap(x => x)
      } else {
        byScore.transform(x => x.setScore(x.row.getDouble(0)))
        util.Arrays.sort(byScore, byScoreComparator)

        iterateLabels(Seq(), byScore, byLabel)
      }).map(Row.fromSeq)
    }

    override def dataType: DataType = ArrayType(
      StructType(
        (if (isScoreVector) Seq(StructField("scoreIndex", IntegerType)) else Seq()) ++
          (if (isLabelVector) Seq(StructField("labelIndex", IntegerType)) else Seq()) ++
          Seq(StructField("metrics", new VectorUDT))))
  }

  case class ScoreLabel(var label: Double, var score: Double, row: Row) {
    def setScore(value: Double) = {
      score = value
      this
    }

    def setLabel(value: Double) = {
      label = value
      this
    }
  }

  class Metric(val name: String, private val func: (Iterable[ScoreLabel], Iterator[ScoreLabel], Iterable[ScoreLabel], Double) => Double) extends Serializable {

    def this(name: String, func: (Iterable[ScoreLabel], Iterator[ScoreLabel]) => Double) =
      this(name, (byLabel: Iterable[ScoreLabel], byScoreReturned: Iterator[ScoreLabel], byScoreAll: Iterable[ScoreLabel], labelRelevanceThreshold: Double) => func(byLabel, byScoreReturned))

    def apply(byLabel: Iterable[ScoreLabel], byScoreReturned: Iterator[ScoreLabel], byScoreAll: Iterable[ScoreLabel], labelRelevanceThreshold: Double): Double = {
      func(byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold)
    }
  }


  private case class AucAccumulator(height: Int, area: Int, positives: Int, negatives: Int)

  def auc(name: String = "auc") = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    val accumulator = byScoreAll.foldLeft(AucAccumulator(0, 0, 0, 0))((accumulated, current) => {
      if (current.label >= labelRelevanceThreshold) {
        accumulated.copy(height = accumulated.height + 1, positives = accumulated.positives + 1)
      } else {
        accumulated.copy(area = accumulated.area + accumulated.height, negatives = accumulated.negatives + 1)
      }
    })

    val denomintor = accumulator.negatives * accumulator.height
    if (denomintor == 0) {
      Double.NaN
    } else {
      accumulator.area.toDouble / denomintor
    }
  })

  def numPositives(name: String = "numPositives") = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byLabel.count(_.label >= labelRelevanceThreshold)
  })

  def foundPositives(name: String = "foundPositives") = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byScoreReturned.count(_.label >= labelRelevanceThreshold)
  })

  def countIf(name: String, filter: Row => Boolean) = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byScoreReturned.count(x => filter(x.row))
  })

  def countIfAt(name: String, size: Int, filter: Row => Boolean) = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byScoreReturned.take(size).count(x => filter(x.row))
  })

  def countRelevantIf(name: String, filter: Row => Boolean) = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byScoreReturned.count(x => x.label >= labelRelevanceThreshold && filter(x.row))
  })

  def countRelevantIfAt(name: String, size: Int, filter: Row => Boolean) = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byScoreReturned.take(size).count(x => x.label >= labelRelevanceThreshold && filter(x.row))
  })

  def countDistinctIf[T](name: String, filter: Row => Boolean, extractor: Row => T) = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byScoreReturned.filter(x => filter(x.row)).map(x => extractor(x.row)).toSet.size
  })

  def countDistinctRelevantIf[T](name: String, filter: Row => Boolean, extractor: Row => T) = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byScoreReturned.filter(x => x.label >= labelRelevanceThreshold && filter(x.row)).map(x => extractor(x.row)).toSet.size
  })

  def countDistinctIfAt[T](name: String, size: Int, filter: Row => Boolean, extractor: Row => T) = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byScoreReturned.take(size).filter(x => filter(x.row)).map(x => extractor(x.row)).toSet.size
  })

  def countDistinctRelevantIfAt[T](name: String, size: Int, filter: Row => Boolean, extractor: Row => T) = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byScoreReturned.take(size).filter(x => x.label >= labelRelevanceThreshold && filter(x.row)).map(x => extractor(x.row)).toSet.size
  })

  def numNegatives(name: String = "numNegatives") = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byLabel.count(_.label < labelRelevanceThreshold)
  })

  def foundNegatves(name: String = "foundNegatives") = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    byScoreReturned.count(_.label < labelRelevanceThreshold)
  })

  def precision(name: String = "precision") = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    if (byScoreReturned.nonEmpty) {
      var size: Int = 0
      var relevant: Int = 0
      byScoreReturned.foreach(i => {
        size += 1
        if (i.label >= labelRelevanceThreshold) {
          relevant += 1
        }
      })
      relevant.toDouble / size.toDouble
    } else if (byLabel.count(_.label >= labelRelevanceThreshold) > 0) 0.0 else Double.NaN
  })

  def recall(name: String = "recall") = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    val allPositive: Int = byLabel.count(_.label >= labelRelevanceThreshold)
    if (allPositive > 0) byScoreReturned.count(_.label >= labelRelevanceThreshold).toDouble / allPositive else Double.NaN
  })

  def precisionAt(at: Int, name: Option[String] = None) = new Metric(name.getOrElse(s"precisionAt$at"), (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    precision()(byLabel, byScoreReturned.take(at), byScoreAll, labelRelevanceThreshold)
  })

  def recallAt(at: Int, name: Option[String] = None) = new Metric(name.getOrElse(s"recalAt$at"), (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    recall()(byLabel.take(at), byScoreReturned.take(at), byScoreAll, labelRelevanceThreshold)
  })

  def f1(name: String = "f1") = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    var truePos: Int = 0
    var allRecommended: Int = 0

    byScoreReturned.foreach(i => {
      allRecommended += 1
      if (i.label >= labelRelevanceThreshold) {
        truePos += 1
      }
    })

    val allPositive: Int = byLabel.count(_.label >= labelRelevanceThreshold)
    if (allPositive > 0) 2.0 * truePos / (allRecommended + allPositive) else Double.NaN
  })

  def f1At(at: Int, name: Option[String] = None) = new Metric(name.getOrElse(s"f1At$at"), (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    f1()(byLabel.take(at), byScoreReturned.take(at), byScoreAll, labelRelevanceThreshold)
  })

  def ndcgWeak(name: String = "ndcgWeak") = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    relativeSortedMetric(byLabel, byScoreAll.iterator,
      (accumulated, current) => {
        accumulated + current._1.label / (if (current._2 == 0) 1 else log2(current._2 + 1))
      })
  })

  def ndcgWeakAt(at: Int, name: Option[String] = None) = new Metric(name.getOrElse(s"ndcgWeakAt$at"), (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    ndcgWeak()(byLabel.take(at), byScoreAll.iterator.take(at), byScoreAll.take(at), labelRelevanceThreshold)
  })

  def ndcgStrong(name: String = "ndcg") = new Metric(name, (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    relativeSortedMetric(byLabel, byScoreAll.iterator,
      (accumulated, current) => {
        accumulated + (Math.pow(2, current._1.label) - 1) / log2(current._2 + 2)
      })
  })

  def ndcgStrongAt(at: Int, name: Option[String] = None) = new Metric(name.getOrElse(s"ndcgAt$at"), (byLabel, byScoreReturned, byScoreAll, labelRelevanceThreshold) => {
    ndcgStrong()(byLabel.take(at), byScoreAll.iterator.take(at), byScoreAll.take(at), labelRelevanceThreshold)
  })

  def relativeSortedMetric(byLabel: Iterable[ScoreLabel], byScoreAll: Iterator[ScoreLabel], op: (Double, (ScoreLabel, Int)) => Double, initial: Double = 0.0): Double = {
    val ideal = byLabel.zipWithIndex.foldLeft(initial)(op)
    val actual = byScoreAll.zipWithIndex.foldLeft(initial)(op)

    if (ideal == 0) Double.NaN else actual / ideal
  }
}
