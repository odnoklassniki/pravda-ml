package org.apache.spark.ml.odkl.texts

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, expr, udf}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
  * Created by eugeny.malyutin on 06.05.16.
  * implementation of continuously EWMA/EWMVar(Exponential-Weighted)/Sig(~=z-score) updater as mllib Transformer
  * Term, newFreq, oldEWMA,oldEWMVar -> EWStruct(newEWMA, newEWMVar, Sig)
  *
  * Formulas comes from here: T. Finch. Incremental calculation of weighted mean and variance.
  * Technical report, University of Cambridge, 2009.
  **/
class EWStatsTransformer(override val uid: String) extends Transformer with Params with DefaultParamsWritable {

  val alpha = new DoubleParam(this, "alphaEW", "alpha EW Stats Continous update parameter", ParamValidators.inRange(0.0, 1.0, false, false))
  val beta = new DoubleParam(this, "betaEWStats", "beta EWstats noisy parameter", ParamValidators.inRange(0.0, 1.0, false, false))
  val oldEWStructColName = new Param[String](this, "OldEWStructColName", "column name with old EWStruct")
  val newEWStructColName = new Param[String](this, "NewEWStructColName", "column name with new EWStruct to create")
  val inputFreqColName = new Param[String](this, "InputFreqColName", "column name with Frequency")
  val inputTermColName = new Param[String](this, "termColName", "column with 'Term'-id to distinct items")
  val inputOldTermColName = new Param[String](this, "oldTermColName", "column with 'old_Term'-id to distinct items")
  val newWordsEWMA = new Param[String](this, "EwmaForNVL", "ewma by a previous value for new words - appeared in current day but not presented in previous")
  val timestmapColName = new Param[String](this, "TimestmapColName", "with Timestamp")
  val oldTimestmapColName = new Param[String](this, "oldTimestmapColName", "old Timestamp")

  def this() = this(Identifiable.randomUID("ewStatsTransformer"))

  override def transform(dataset: DataFrame): DataFrame = {

    val preparedDataset = {
      val beforeTimestamp = dataset
        .withColumn($(inputTermColName), expr("NVL(" + $(inputTermColName) + " , " + $(inputOldTermColName) + ")")) //null terms replacement( if there were no term in last day)
        .na.fill(0.0, Array($(inputFreqColName)))

      if (isSet(timestmapColName)){
        beforeTimestamp.withColumn($(timestmapColName), expr("NVL(" + $(timestmapColName) + " , " + $(oldTimestmapColName) + ")"))
      }else{
        beforeTimestamp
      }
    }

    dayStatsComputing(preparedDataset)
  }

  def dayStatsComputing(dfJoinedOld: DataFrame //one day computing
                       ): DataFrame = {

    val alphaD = $(alpha)
    val betaD = $(beta)
    val udfComputeStatistics = udf((term: String, new_freq: Double, old_ewma: Double, old_ewmvar: Double) =>
      EWStatsTransformer.termEWStatsComputing(term, new_freq, old_ewma, old_ewmvar, alphaD, betaD)
    )

    dfJoinedOld
      .withColumn($(newEWStructColName),
        udfComputeStatistics(
          col($(inputTermColName)),
          col($(inputFreqColName)),
          expr("NVL(" + $(oldEWStructColName) + ".ewma," + $(newWordsEWMA) + ")"),
          expr("NVL(" + $(oldEWStructColName) + ".ewmvar,0.0)")
        ))
  }


  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    schema
      .add($(oldEWStructColName), DoubleType)
      .add(StructField($(newEWStructColName),
        StructType(
          StructField("sig", DoubleType, false) ::
            StructField("ewma", DoubleType, false) ::
            StructField("ewmvar", DoubleType, false) :: Nil
        ), true))
  }


  /** @group setParam */
  def setAlpha(value: Double): this.type = set(alpha, value)

  /** @group getParam */
  def getAlpha: Double = $(alpha)

  /** @group setParam */
  def setBeta(value: Double): this.type = set(beta, value)

  /** @group getParam */
  def getBeta: Double = $(beta)

  /** @group setParam */
  def setInputFreqColName(value: String): this.type = set(inputFreqColName, value)

  /** @group setParam */
  def setOldEWStructColName(value: String): this.type = set(oldEWStructColName, value)

  /** @group setParam */
  def setNewEWStructColName(value: String): this.type = set(newEWStructColName, value)

  /** @group setParam */
  def setInputTermColName(value: String): this.type = set(inputTermColName, value)

  /** @group setParam */
  def setOldTermColName(value: String): this.type = set(inputOldTermColName, value)

  /** @group setParam */
  def setTimestampColName(value: String): this.type = set(timestmapColName, value)

  /** @group setParam */
  def setOldTimestampColName(value: String): this.type = set(oldTimestmapColName, value)

  /** @group setParam */
  def setNewWordsEWMA(value: String): this.type = set(newWordsEWMA, value)


  setDefault(
    new ParamPair[Double](beta, 1E-7),
    new ParamPair[Double](alpha, 0.3),
    new ParamPair[String](inputFreqColName, "Freq"),
    new ParamPair[String](inputTermColName, "Term"),
    new ParamPair[String](inputOldTermColName, "old_Term"),
    new ParamPair[String](oldEWStructColName, "old_EWStruct"),
    new ParamPair[String](newEWStructColName, "EWStruct"),
    new ParamPair[String](newWordsEWMA, "0.0"))

}

object EWStatsTransformer extends DefaultParamsReadable[EWStatsTransformer] {

  override def load(path: String): EWStatsTransformer = super.load(path)

  def termEWStatsComputing(term: String, newFreq: Double, oldEWMA: Double, oldEWMVar: Double, alpha: Double, beta: Double) = {
    val delta = newFreq - oldEWMA
    val new_ewma = oldEWMA + alpha * delta
    val new_emwvar = (1 - alpha) * (oldEWMVar + alpha * delta * delta)
    val new_sig = math.max((newFreq - math.max(oldEWMA, beta)) / (math.sqrt(oldEWMVar) + beta), 0)

    EWStruct(new_sig, new_ewma, new_emwvar)
  }

  case class EWStruct(sig: Double, ewma: Double, ewmvar: Double)
}