package org.apache.spark.ml.odkl

import java.util.concurrent.ThreadLocalRandom

import odkl.analysis.spark.util.RDDOperations._
import odkl.analysis.spark.util.collection.CompactBuffer
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, functions}


/**
  * Follows ideas from Combined Regression and Ranking paper (http://www.decom.ufop.br/menotti/rp122/sem/sem2-alex-art.pdf)
  *
  * Can model pair-wise ranking task (sample pairs, substract features and label 1/0), can model point-wise regression,
  * or can combine both by choosing whenever to sample single item or a pair. 
  */
trait CRRSamplerParams extends HasInputCol with HasGroupByColumns with HasLabelCol {

  val groupSampleRate = new DoubleParam(this, "groupSampleRate",
    "The percentage of lists (users) to keep in the taken sample.", (x: Double) => x > 0 && x <= 1)
  val itemSampleRate = new DoubleParam(this, "itemSampleRate",
    "The percentage of items to keep in the sample (comparing to the amount of items, not to amount of possible pairs).", (x: Double) => x > 0 && x <= 1)
  val rankingPower = new DoubleParam(this, "rankingPower",
    "The percentage of paired sample (for each selected item we decide whenever to return it as is or to select a pair.", (x: Double) => x >= 0 && x <= 1)
  val shuffleToPartitions = new IntParam(this, "shuffleToPartitions",
    "Since after sampling the size might decrease, it is worth to shuffle dataset to fewer partitions. Shuffle also can improve" +
      "convergence by providing better order randomization.", (x: Int) => x > 0)

  def setGroupSampleRate(value: Double): this.type = set(groupSampleRate, value)

  def setItemSampleRate(value: Double): this.type = set(itemSampleRate, value)

  def setRankingPower(value: Double): this.type = set(rankingPower, value)

  def setShufflerToPartitions(value: Int) : this.type = set(shuffleToPartitions, value)

  setDefault(
    rankingPower -> 0.0,
    inputCol -> "features",
    labelCol -> "label"
  )
}

/**
  * Model applied as a transformer, but the resulting data set is not determenistic (each pass produces different
  * results). Results must not be cached.
  */
class CRRSamplerModel(override val uid: String) extends
  Model[CRRSamplerModel] with DefaultParamsWritable with CRRSamplerParams with HasNetlibBlas{

  def this() = this(Identifiable.randomUID("crrSampler"))

  setDefault(
    groupSampleRate -> 1.0,
    itemSampleRate -> 1.0
  )

  override def transform(dataset: DataFrame): DataFrame = {

    val (data, keyIndex, toDrop) = if (isDefined(groupByColumns) && $(groupByColumns).length > 0) {
      if ($(groupByColumns).length == 1) {
        (dataset, dataset.schema.fieldIndex($(groupByColumns).head), None)
      } else {
        val keyStruct = functions.struct($(groupByColumns).map(x => dataset(x)): _*)
        val keyName = s"${uid}_tmpKey"
        val data = dataset.withColumn(keyName, keyStruct)

        (data, data.schema.fieldIndex(keyName), Some(keyName))
      }
    } else {
      val keyName = s"${uid}_tmpKey"
      val data = dataset.withColumn(keyName, functions.lit(keyName))

      (data, data.schema.fieldIndex(keyName), Some(keyName))
    }

    val featuresIndex = data.schema.fieldIndex($(inputCol))
    val labelIndex = data.schema.fieldIndex($(labelCol))


    val result = dataset.sqlContext.createDataFrame(
      data
        .rdd
        .groupWithinPartitionsBy(x => x.get(keyIndex))
        .flatMap(x => sampleRows(x._2, labelIndex, featuresIndex)),
      data.schema
    )

    val noExtraColumn = toDrop.map(x => result.drop(x)).getOrElse(result)

    get(shuffleToPartitions).map(x => noExtraColumn.repartition(x)).getOrElse(noExtraColumn)
  }

  def sampleRows(rows: Iterator[Row], labelIndex: Int, featureIndex: Int): Iterator[Row] = {
    if ($(groupSampleRate) < 1 && ThreadLocalRandom.current().nextDouble() > $(groupSampleRate)) {
      // Skip entire group
      Iterator.empty
    } else if ($(rankingPower) > 0) {

      val positives = new CompactBuffer[Row]()
      val negatives = new CompactBuffer[Row]()

      rows.foreach(x => if (x.getDouble(labelIndex) > 0) {
        positives += x
      } else {
        negatives += x
      })

      if (positives.isEmpty || negatives.isEmpty) {
        // Could not use single class groups when ranking part is enabled.
        Iterator.empty
      } else {
        counterSample(positives.iterator, negatives, featureIndex) ++ counterSample(negatives.iterator, positives, featureIndex)
      }
    } else {
      counterSample(rows, Seq.empty, featureIndex)
    }
  }

  def counterSample(source: Iterator[Row], counterPart: Seq[Row], featureIndex: Int): Iterator[Row] = {
    val sampled = if ($(itemSampleRate) < 1) {
      source.filter(_ => ThreadLocalRandom.current().nextDouble() < $(itemSampleRate))
    } else {
      source
    }

    if ($(rankingPower) <= 0) {
      // Pure regression
      sampled
    } else if ($(rankingPower) < 1) {
      // Pure ranking
      sampled.map(x => pairSample(x, counterPart, featureIndex))
    } else {
      // Mixture
      sampled.map(x => {
        if (ThreadLocalRandom.current().nextDouble() < $(rankingPower)) {
          pairSample(x, counterPart, featureIndex)
        } else {
          x
        }
      })
    }
  }

  def pairSample(sample: Row, counterPart: Seq[Row], featureIndex: Int): Row = {
    val counterSample: Vector = counterPart(ThreadLocalRandom.current().nextInt(counterPart.size)).getAs[Vector](featureIndex)

    val result = Array.tabulate(sample.length) { i =>
      if (i == featureIndex) {
        val features = sample.getAs[Vector](featureIndex)

        val result = Vectors.zeros(features.size).toDense

        axpy(1.0, features, result.values)
        axpy(-1.0, counterSample, result.values)

        result
      } else {
        sample.get(i)
      }
    }

    Row.fromSeq(result)
  }

  override def copy(extra: ParamMap): CRRSamplerModel = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema
}

/**
  * Estimator is used to select the proper item sample rate to achive desired size of the resulting
  * sample. Takes into consideration the source dataset size and the amount of valid for ranking lists
  * (list with samples of different rank).
  */
class CRRSamplerEstimator(override val uid: String) extends Estimator[CRRSamplerModel]
  with DefaultParamsWritable with CRRSamplerParams {

  val expectedNumSamples = new IntParam(this, "expectedNumSamples",
    "The expected number of samples in the result. Required.", (x: Int) => x > 0)

  def setExpectedNumberOfSamples(value: Int) : this.type = set(expectedNumSamples, value)

  setDefault(
    groupSampleRate -> 1.0
  )

  def this() = this(Identifiable.randomUID("crrSamplerEstimator"))

  override def fit(dataset: DataFrame): CRRSamplerModel = {
    val totalSamples: Double = if ($(rankingPower) > 0 && isDefined(groupByColumns) && $(groupByColumns).length > 0) {
      val (withKey, keyIndex) = if ($(groupByColumns).length == 1) {
        (dataset, dataset.schema.fieldIndex($(groupByColumns).head))
      } else {
        val key = functions.struct($(groupByColumns).map(x => dataset(x)): _*)
        val keyName = s"${uid}_tmpKey"
        val data = dataset.withColumn(keyName, key)

        (data, data.schema.fieldIndex(keyName))
      }

      val labelIndex = withKey.schema.fieldIndex($(labelCol))

      withKey
        .rdd
        .groupWithinPartitionsBy(x => x.get(keyIndex))
        .map(x => {
          var numPositives: Long = 0
          var numNegatives: Long = 0
          x._2.foreach(row => if (row.getDouble(labelIndex) > 0) numPositives += 1 else numNegatives += 1)

          if(numNegatives > 0 && numPositives > 0) numNegatives + numPositives else 0
        }).sum()
    } else {
      dataset.count()
    }

    val discountedByGroupRate = totalSamples * $(groupSampleRate)

    val requiredItemSampleRate = Math.min(1.0, $(expectedNumSamples) / discountedByGroupRate)

    logInfo(s"Estimated total number of samples $totalSamples, after groups sampling $discountedByGroupRate. Chosen item sample rate is $requiredItemSampleRate")

    val model: CRRSamplerModel = copyValues(new CRRSamplerModel(), ParamMap(itemSampleRate -> requiredItemSampleRate))
    model.setParent(this)
  }

  override def copy(extra: ParamMap): CRRSamplerEstimator = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = schema
}