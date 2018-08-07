package org.apache.spark.ml.odkl.texts

import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{ArrayType, StringType, StructType}
import org.apache.spark.util.collection.OpenHashMap

import scala.collection.Map

/**
  * Created by eugeny.malyutin on 06.05.16.
  *
  * Original CountVectorizer is badly implemented - does not conform to ML pipeline interface, uses caching
  * in a leaking manner and etc. Here we fix part of the problems, but more is to be done.
  *
  * TODO: Get rid of Vocabulary in constructor, adapt o ModelWithSummary interface.
  **/
trait OdklCountVectorizerParams extends Params {
  val vocabAttrGroupName = new Param[String](this, "vocabAttrGroupName", "name for AttrGroup with vocab (or size) to store in MetadData")

  /** @group getParam */
  def getVocabAttrGroupName() = $(vocabAttrGroupName)

  /** @group setParam */
  def setVocabAttrGroupName(value: String): this.type = set(vocabAttrGroupName, value)

  val storeVocabInMetadata = new BooleanParam(this, "storeVocabInMetadata", "if true vocab will be stored as AttributeGroup(may cause GC overhead later)\" +\n      \", else only vocabSize will be stored(default)")

  /** @group getParam */
  def getStoreVocabInMetadata() = $(storeVocabInMetadata)

  /** @group setParam */
  def setStoreVocabInMetadata(value: Boolean): this.type = set(storeVocabInMetadata, value)

  setDefault(new ParamPair[String](vocabAttrGroupName, "vocab"), new ParamPair[Boolean](storeVocabInMetadata, false))

}

/**
  * ml.feature.CountVectorizer and CountVectorizerModel extension with vocabulary or vocab size saved
  * in outputColumn metadata as AttributeGroup
  *
  * @param uid
  * @param vocabulary BagOfWords vocabulary from mllib.CountVectorizer
  **/
class OdklCountVectorizerModel(override val uid: String, override val vocabulary: Array[String])
  extends CountVectorizerModel(uid, vocabulary) with OdklCountVectorizerParams {
  //change vocabulary to StringArrayParam when ApacheSpark fixes this

  override def transform(dataset: Dataset[_]): DataFrame = {
    val supDF = super.transform(dataset)
    supDF.withColumn($(outputCol), supDF.col($(outputCol)).as($(outputCol), metadataToAdd(vocabulary)))
  }

  def metadataToAdd(vocab: Array[String]) = {
    {
      if ($(storeVocabInMetadata)) {
        val defaultAttr = NumericAttribute.defaultAttr
        new AttributeGroup($(vocabAttrGroupName), vocab.map(defaultAttr.withName).asInstanceOf[Array[Attribute]])
      } else {
        val defaultAttr = NumericAttribute.defaultAttr
        new AttributeGroup($(vocabAttrGroupName), vocab.length)
      }
    }.toMetadata()
  }

  override def transformSchema(schema: StructType) = {

    SchemaUtils.checkColumnType(schema, $(inputCol), new ArrayType(StringType, true))

    schema.add($(outputCol), new VectorUDT, false, metadataToAdd(vocabulary))
  }
}

class OdklCountVectorizer(override val uid: String) extends CountVectorizer(uid) with OdklCountVectorizerParams {

  def this() = this(Identifiable.randomUID("odklCountVectorizer"))

  val inheritedVocabulary = new Param[Map[String,Int]](
    this, "inheritedVocabulary", "Dictionary inherited from the previous epoche. Can be used to try to preserve word indices.")

  override def fit(dataset: Dataset[_]): CountVectorizerModel = {
    transformSchema(dataset.schema, logging = true)
    val vocSize = $(vocabSize)
    val input = dataset.select($(inputCol)).rdd.map(_.getAs[Seq[String]](0))
    val minDf = if ($(minDF) >= 1.0) {
      $(minDF)
    } else {
      $(minDF) * input.count()
    }
    val wordCounts: RDD[(String, Long)] = input.flatMap { case (tokens) =>
      val wc = new OpenHashMap[String, Long]
      tokens.foreach { w =>
        wc.changeValue(w, 1L, _ + 1L)
      }
      wc.map { case (word, count) => (word, (count, 1)) }
    }.reduceByKey { case ((wc1, df1), (wc2, df2)) =>
      (wc1 + wc2, df1 + df2)
    }.filter { case (word, (wc, df)) =>
      df >= minDf
    }.map { case (word, (count, dfCount)) =>
      (word, count)
    }

    val vocab: Array[String] = wordCounts
      .sortBy(_._2, ascending = false, numPartitions = 1)
      .map(_._1)
      .take(vocSize)

    require(vocab.length > 0, "The vocabulary size should be > 0. Lower minDF as necessary.")

    val mayBeMerged: Array[String] = if (isDefined(inheritedVocabulary)) {
      val newVocabulary = new Array[String](vocab.length)

      val previousMap: Map[String, Int] = $(inheritedVocabulary)

      val remainder = new scala.collection.mutable.Queue[String]()

      for(word <- vocab) {
         previousMap.get(word) match {
           case index : Some[Int] => newVocabulary(index.get) = word
           case _ => remainder.enqueue(word)
         }
      }

      for(i <- newVocabulary.indices) {
        if (newVocabulary(i) == null) {
          newVocabulary(i) = remainder.dequeue()
        }
      }

      newVocabulary
    } else {
      vocab
    }

    copyValues(new OdklCountVectorizerModel(Identifiable.randomUID("odklCountVectorizerModel"), mayBeMerged).setParent(this))
  }
}



