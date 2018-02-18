package org.apache.spark.ml.odkl.texts

import java.util.Random

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol, HasSeed}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.linalg.{Matrices, SparseMatrix, Vector}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{LongType, StructType}

/**
  * Created by eugeny.malyutin on 05.05.16.
  *
  *  Implementation of Locality-sensitive hashing(similar vectors - similar hashes) via Random Binary Projection as ml.Transformer
  * requires DataFrame with inputCol as linalg.Vector as data representation and output's Long column with HashValue
  *
  * If dimensions is not set - will search  AttributeGroup in metadata
  */
class RandomProjectionsHasher(override val uid: String) extends Transformer
 with HasInputCol with HasOutputCol with HasSeed{


  val basisSize =
    new LongParam(this, "basisSize", "number of random vector-normales e.g. 2 ^ binsNum buckets", ParamValidators.gt(0))

  val dim =
    new LongParam(this, "dim", "dimension of sparse vectors e.g. Bag-Of-Word vocab size")

  val sparsity = new DoubleParam(this,"sparsity","sparsity param in mllib.linal.Matrices.sparndn",ParamValidators.inRange(0.0,1.0,false,true))

  /** @group getParam */
  def getSparsity: Double = $(sparsity)

  /** @group getParam */
  def getBasisSize: Long = $(basisSize)

  /** @group getParam */
  def getDim: Long = $(dim)


  setDefault(new ParamPair[Long](basisSize, 18L), new ParamPair[Double](sparsity,0.5))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setSparsity(value: Double): this.type = set(sparsity, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setBasisSize(value: Long): this.type = set(basisSize, value)

  /** @group setParam */
  def setDim(value: Long): this.type = set(dim, value)


  def this() = this(Identifiable.randomUID("randomProjectionsHasher"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val dimensity = {
      if (!isSet(dim)) {//If dimensions is not set - will search  AttributeGroup in metadata as it comes from OdklCountVectorizer
        val vectorsIndex = dataset.schema.fieldIndex($(inputCol))
        AttributeGroup.fromStructField(dataset.schema.fields(vectorsIndex)).size
      } else {
        $(dim).toInt
      }
    }
    val projectionMatrix = dataset.sqlContext.sparkContext.broadcast(
      Matrices.sprandn($(basisSize).toInt, dimensity, $(sparsity), new Random($(seed))).asInstanceOf[SparseMatrix])
  //the matrix of random vectors to costruct hash

    val binHashSparseVectorColumn = udf((vector: Vector) => {
      projectionMatrix.value.multiply(vector).values
        .map(f =>  if (f>0) 1L else 0L)
        .view.zipWithIndex
        .foldLeft(0L) {case  (acc,(v, i)) => acc | (v << i) }

    })
    dataset.withColumn($(outputCol), binHashSparseVectorColumn(dataset.col($(inputCol))))
  }

  override def copy(extra: ParamMap): Transformer = {
    defaultCopy(extra)
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.appendColumn(schema, $(outputCol), LongType)
  }

}
