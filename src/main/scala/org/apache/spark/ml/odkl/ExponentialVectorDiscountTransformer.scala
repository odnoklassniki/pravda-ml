package org.apache.spark.ml.odkl

import odkl.analysis.spark.util.IteratorUtils
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}


/**
  * Created by eugeny.malyutin on 18.02.18.
  *
  * Transformer to implement exponential weighted discounting for vectors;
  * Expects dataFrame with structure ( $"groupByColumns", $"timestamp", $"vector")
  *
  * Return dataFrame  ( $"groupByColumns", $"timestamp", $"vector)
  *
  * $"timestamp" - last seen action timestamp for this $"identificator"
  * $"vector" - summed actions. vector(0) is reserved for "aggregation" timestamp
  *
  */
class ExponentialVectorDiscountTransformer(override val uid: String)
    extends Transformer with DefaultParamsWritable with HasGroupByColumns {

  val timestampColumn: Param[String] =  new Param[String](this, "timestampCol", "timestamp column")
  val vectorColumn: Param[String] =  new Param[String](this, "vectorCol", "column with vector")

  val exponentBase: DoubleParam =  new DoubleParam(this, "expBase", "exponent base parameter", ParamValidators.inRange(0.0, 1.0))
  val exponentScale: LongParam =  new LongParam(this, "timeScaling", "time scaler", ParamValidators.gt(0.0))
  val timeNow: LongParam =  new LongParam(this, "timeNow", "time to discount vectors at", ParamValidators.gt(0))

  val vectorsSize: IntParam =  new IntParam(this, "vectorSize", "size of a vector", ParamValidators.gt(0))

  val numPartitions: IntParam =  new IntParam(this, "numPartitions", "num partitions", ParamValidators.gt(0))




  /** @group setParam */
  def setTimestampColumn(value: String): this.type = set(timestampColumn, value)

  /** @group setParam */
  def setVectorColumn(value: String): this.type = set(vectorColumn, value)


  /** @group setParam */
  def setExponentBase(value: Double): this.type = set(exponentBase, value)

  /** @group setParam */
  def setExponentScale(value: Long): this.type = set(exponentScale, value)

  /** @group setParam */
  def setTimeNow(value: Long): this.type = set(timeNow, value)

  /** @group setParam */
  def setVectorsSize(value: Int): this.type = set(vectorsSize, value)

  /** @group setParam */
  def setNumPartitions(value: Int): this.type  = set(numPartitions, value)

  /**
    * Perform aggregation with repartition-and-sortWithinPartitions style
    *
    * 1) repartition dataFrame by repartitionBy columns
    * 2) Sorts dataframe via sortColumns + timestamp.asc
    * 3) all data about vectors for identificator - is in one partition and sorted by timestamp;
    *   Iterate through partition and aggregate it
    * 4) Map RDD back to dataFrame
    */
  override def transform(dataset: Dataset[_]) = {
    val dummyVector =  {
      Vectors.sparse($(vectorsSize), Seq(0 -> $(timeNow).toDouble))
    }

    val resultRDD = dataset
      .repartition($(numPartitions), $(groupByColumns).map(c => dataset.col(c)): _*)
      .sortWithinPartitions($(groupByColumns).map(dataset.col) :+ dataset.col($(timestampColumn)).asc: _*)
      .toDF.rdd
      .mapPartitions(it => {
        val mapped: Iterator[(Seq[Any], (Long, Vector))] = it
          .map(row => $(groupByColumns).toSeq.map(row.getAs[Any]) ->
            (row.getAs[Long]($(timestampColumn)), row.getAs[Vector]($(vectorColumn))))

        IteratorUtils.groupByKey[Seq[Any], (Long, Vector)](mapped)
          .map {
            case (key, data) => {
              val result =
                data.reduce((left, right) => right._1 -> discountVectors(left._2, right._2, $(exponentBase), $(exponentScale), right._2(0).toLong))

              key ++
                Seq(result._1,
                  discountVectors(result._2, dummyVector, $(exponentBase), $(exponentScale), $(timeNow))) //discount last action to timeNow timestamp
            }
          }
      })
      .map(Row.fromSeq(_))

    dataset.sqlContext.createDataFrame(resultRDD, transformSchema(dataset.schema))

  }

  override def copy(extra: ParamMap):Transformer = defaultCopy(extra)

  def this() = {
    this(Identifiable.randomUID("exponentialDiscounter"))
  }

  override def transformSchema(schema: StructType) = {
      val neededColumns = ($(groupByColumns) ++ Seq($(timestampColumn), $(vectorColumn)) )

    StructType(schema.filter(s => neededColumns.contains(s.name)))
  }

  /**
    * Function to shrink vectors and discount it with Vector Utils, contract:
    * vector(0) - timestamp
    * vector2(0) should goes after vector1
    * if vectors size differs - pad or shrink vector1 to vector2 size
    * @param vectorOld earlier vector
    * @param vectorNew later vector
    * @param base - base for exponent
    * @param scale - parameter to scale time difference
    * @param timeNow - time to weight vectors to
    * @return result vector
    */
  def discountVectors(vectorOld: Vector, vectorNew: Vector, base: Double, scale: Double, timeNow: Long) = {

    val mayBeUpdatedVectorOld = if (vectorNew.size > vectorOld.size) {
      Vectors.dense(vectorOld.toArray.padTo(vectorNew.size, 0.0))
    } else if (vectorNew.size < vectorOld.size) {
      Vectors.dense(vectorOld.toArray.take(vectorNew.size))
    } else {
      vectorOld
    }

    discount(mayBeUpdatedVectorOld, vectorNew, base, scale, timeNow)
  }
  /**
    * Discount vectors. Expects vector(0) as timestamp
    *
    * @param vector1 earlier vector
    * @param vector2 later vector
    * @param base - base for exponent
    * @param scale - parameter to scale time difference
    * @param timeNow - time to weight vetors to
    * @return result vector
    */
  def discount(vector1: Vector, vector2: Vector, base:Double, scale:Double, timeNow:Long): Vector = {
    val ans = (vector1.asBreeze * Math.pow(base, (timeNow - vector1(0)) / scale)) +
      (vector2.asBreeze * Math.pow(base, (timeNow - vector2(0)) / scale))

    ans(0) = timeNow //update time
    Vectors.fromBreeze(ans).compressed
  }



  setDefault(
    new ParamPair[String](timestampColumn,"timestamp"),
    new ParamPair[String](vectorColumn, "vector"),
    new ParamPair[Double](exponentBase, 0.7),
    new ParamPair[Long](exponentScale, 1000L * 60 * 60 * 24 * 2),
    new ParamPair[Int](numPartitions, 511)
  )
}
