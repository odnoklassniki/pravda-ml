package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utility used fo cross-validation.
  */

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.odkl.ModelWithSummary.Block
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types.{IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext, functions}

import scala.util.Try

/**
  * Used to train and evaluate model in folds.
  *
  * @param nested Nested estimator (it supposed that evaluator is already included).
  */
class CrossValidator[M <: ModelWithSummary[M]]
(
  nested: SummarizableEstimator[M],
  override val uid: String

)
  extends ForkedEstimatorSameType[M, Int](nested, uid) with HasIsTestCol with HasFolds {

  val addGlobal = new BooleanParam(this, "addGlobal", "Whenever to add fold with global data")

  setDefault(addGlobal -> true)

  def setAddGlobal(value: Boolean) : this.type = set(addGlobal, value)

  def this(nested: SummarizableEstimator[M]) = this(nested, Identifiable.randomUID("kFoldEvaluator"))

  override def copy(extra: ParamMap): CrossValidator[M] = {
    copyValues(new CrossValidator[M](nested.copy(extra)), extra)
  }


  override protected def createForks(dataset: Dataset[_]): Seq[(Int, DataFrame)] = {
    val numFoldsValue: Int = getNumFolds(dataset.toDF)

    val folds = for (i <- 0 until numFoldsValue)
      yield (i, dataset.withColumn($(isTestColumn), dataset($(numFoldsColumn)) === i))

    if ($(addGlobal)) {
      folds ++ Seq((-1, dataset.withColumn($(isTestColumn), functions.lit(false))))
    } else {
      folds
    }
  }

  def getNumFolds(dataset: DataFrame): Int = {
    val numFoldsValue = if ($(numFolds) > 0) $(numFolds)
    else {
      dataset.select($(numFoldsColumn)).agg(functions.max($(numFoldsColumn))).take(1).head.getAs[Number](0).intValue()
    }
    numFoldsValue
  }

  override protected def mergeModels(sqlContext: SQLContext, models: Seq[(Int, Try[M])]): M = {
    val trueModels = models.map(x => x._1 -> x._2.get)
    val wholeModel: M = if ($(addGlobal)) trueModels.find(_._1 == -1).get._2 else trueModels.find(_._1 == 0).get._2
    val foldModels = trueModels.map(x => x._1 -> x._2).filter(_._1 >= 0)



    val extendedBlocks = foldModels
      .foldLeft(wholeModel.summary.blocks.mapValues(_.withColumn($(numFoldsColumn), functions.lit(-1))))(
        (agg: Map[Block, DataFrame], model: (Int, M)) => {

          val extended = model._2.summary.blocks.mapValues(_.withColumn($(numFoldsColumn), functions.lit(model._1)))

          agg ++ extended.map(x => (x._1, if (agg.contains(x._1)) x._2.unionAll(agg(x._1)) else x._2))
        })


    wholeModel.copy(extendedBlocks)
  }
}

object CrossValidator extends DefaultParamsReadable[CrossValidator[_]] with Serializable {
  /**
    * Utility used to assign folds to instances. Byt default based on the hash of entire row, but might
    * also use only a sub set of columns.
    */
  class FoldsAssigner(override val uid: String) extends Transformer with HasFolds with DefaultParamsReadable[FoldsAssigner]{
    def this() = this(Identifiable.randomUID("foldsAssigner"))

    val partitionBy = new StringArrayParam(this, "partitionBy", "Columns to partition dataset by")

    def setPartitionBy(columns: String*) = set(partitionBy, columns.toArray)

    override def transform(dataset: Dataset[_]): DataFrame = {
      val partition = SparkSqlUtils.reflectionLock.synchronized(
        if ($(numFolds) > 0) {
          functions.udf[Int, Row](x => Math.abs(x.hashCode() % $(numFolds)))
        } else {
          functions.udf[Int, Row](x => 0)
        })

      val columns = if (isDefined(partitionBy)) {
        $(partitionBy).map(dataset(_))
      }
      else {
        dataset.columns.map(dataset(_))
      }
      dataset.withColumn($(numFoldsColumn), partition(functions.struct(columns: _*)))
    }

    override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

    @DeveloperApi
    override def transformSchema(schema: StructType): StructType = schema.add($(numFoldsColumn), IntegerType, nullable = false)
  }

  object FoldsAssigner extends DefaultParamsReadable[FoldsAssigner]
}
