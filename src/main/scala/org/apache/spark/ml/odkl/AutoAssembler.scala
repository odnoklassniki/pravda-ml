package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utility for extracting columns into feature vectors.
  */

import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, PipelineModel, Transformer}
import org.apache.spark.ml.param.shared.HasOutputCol
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.sql.{DataFrame, Dataset, functions}
import org.apache.spark.sql.types._

/**
  * Params for automatic feature-vector assembler.
  */
trait AutoAssemblerParams extends HasColumnsSets with HasOutputCol with HasColumnAttributeMap {
  def setOutputCol(value: String): this.type = set(outputCol, value)
}

/**
  * Utility for automatically assembling columns into a vector of features. Takes either all the columns, or
  * a subset of them. For boolean, numeric and vector columns uses default vectorising logic, for string and collection
  * columns applies nominalizers.
  *
  * @param uid
  */
class AutoAssembler(override val uid: String) extends Estimator[PipelineModel]
  with AutoAssemblerParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("autoAssembler"))

  override def fit(dataset: Dataset[_]): PipelineModel = {
    val columns = extractColumns(dataset.toDF())

    val nominal: Array[StructField] = columns.filter(
      x => x.dataType.isInstanceOf[StringType]
        || x.dataType.isInstanceOf[ArrayType] && x.dataType.asInstanceOf[ArrayType].elementType.isInstanceOf[StringType])



    val nominalizers: Array[Transformer] = if (nominal.length > 0) {

      val mayBeExploded = nominal.foldLeft(dataset.toDF)((data, field) =>
        if (field.dataType.isInstanceOf[ArrayType])
          data.withColumn(field.name, functions.explode(data(field.name)))
        else data)

      val expressions = nominal.map(x => SQLOperations.collectAsSet(StringType)(mayBeExploded(x.name)).as(x.name))
      val values = mayBeExploded.groupBy().agg(expressions.head, expressions.drop(1) : _*).collect()

      require(!values.isEmpty, s"Could not extract nominal values from empty dataset at $uid")
      
      nominal.zipWithIndex.map(x =>
        new MultinominalExtractorModel()
          .setInputCol(x._1.name)
          .setOutputCol(x._1.name)
          .setValues(values(0).getAs[Seq[String]](x._1.name).sorted : _*))
    }
    else {
      Array()
    }

    new PipelineModel(
      Identifiable.randomUID("autoAssemblerPipeline"),
      nominalizers ++
        Array[Transformer](
          new NullToNaNVectorAssembler().setInputCols(
            columns.map(_.name))
            .setOutputCol($(outputCol))
            .setColumnAttributeMap($(columnAttributeMap).toSeq :_*)
        )).setParent(this)
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    val transformed = schema.fields.map {
      case unchanged: StructField
        if unchanged.dataType.isInstanceOf[NumericType]
          || unchanged.dataType.isInstanceOf[BooleanType]
          || unchanged.dataType.isInstanceOf[VectorUDT] => unchanged
      case nominalized: StructField
        if nominalized.dataType.isInstanceOf[StringType]
          || nominalized.dataType.isInstanceOf[ArrayType] && nominalized.dataType.asInstanceOf[ArrayType].elementType.isInstanceOf[StringType]
      => nominalized.copy(dataType = new VectorUDT)
      case leftOver: StructField => leftOver
    }

    new StructType(transformed :+ new StructField($(outputCol), new VectorUDT, true))
  }
}

/**
  * Adds read logic
  */
object AutoAssembler extends DefaultParamsReadable[AutoAssembler]
