package org.apache.spark.ml.odkl.texts

import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.Vectors.norm
import org.apache.spark.mllib.linalg.{BLAS, Vector}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by eugeny.malyutin on 05.05.16.
  *
  *  Deduplicator based on [[RandomProjectionsHasher]] and cosine similarity.
  *
  **/

class HashBasedDeduplicator(override val uid: String) extends Transformer with Params with Logging {

  val similarityThreshold =
    new DoubleParam(this, "simTresh",
      "cosine similarity Treshold for dedupolication in one hash-bucket for vectors to be marked as 'similar' " +
        "\n 0.9 by default",
      ParamValidators.inRange(0.0,1.0,false,true))//in (0;1]

  val inputColHash = new Param[String](this, "inputColHash",
    "column with LSH(local sensitive hashing) as Long \n \"hash\" by default")

  val inputColVector = new Param[String](this, "inputColVector",
    "column with Vector data representation")


  /** @group setParam */
  def setInputColVector(value: String): this.type = set(inputColVector, value)

  /** @group setParam */
  def setInputColHash(value: String): this.type = set(inputColHash, value)

  /** @group setParam */
  def setSimilarityTreshold(value: Double): this.type = set(similarityThreshold, value)

  setDefault(new ParamPair[String](inputColHash,"hash"),
    new ParamPair[Double](similarityThreshold,0.9))

  def this() = this(Identifiable.randomUID("hashBasedDeduplication"))

  override def transform(dataset: DataFrame): DataFrame = {
    dataset.sqlContext.createDataFrame(
      dataset
        .repartition(dataset.col($(inputColHash)))
        .sortWithinPartitions($(inputColHash))
        .rdd
        .mapPartitions((f: Iterator[Row]) => {
          if (f.hasNext) {
            var curHash: Long = -1L
            val vectorsBuffer = new ArrayBuffer[Vector](0) // unique vectors buffer for this bucket
            for (it <- f) yield {
              val newHash = it.getAs[Long]($(inputColHash))
              if (newHash == curHash) {
                val currentVector = it.getAs[Vector]($(inputColVector))
                val isUnique = vectorsBuffer.forall(storedVector => { //are this vector is "different" with other in buffer?
                  ((BLAS.dot(storedVector, currentVector)) / (norm(storedVector, 2) * norm(currentVector, 2))) < $(similarityThreshold) //is unsimilar?
                })
                if (isUnique) {
                  vectorsBuffer.append(currentVector)
                  it
                } else {
                  Row.empty //dummy Row
                }
              } else {
                vectorsBuffer.clear()
                vectorsBuffer.append(it.getAs[Vector]($(inputColVector)))
                curHash = newHash
                it
              }
            }
          } else {
            new Array[Row](0).toIterator //empty partition?
          }

        }).filter(!_.equals(Row.empty)), //filter dummy
      transformSchema(dataset.schema))
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)


}
