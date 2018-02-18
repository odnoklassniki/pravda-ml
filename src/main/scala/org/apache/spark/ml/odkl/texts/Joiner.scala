package org.apache.spark.ml.odkl.texts

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}

/**
  * Created by eugeny.malyutin on 06.05.16.
  *
  * dataframe's join as transformer with right dataframe and col. expression as parameters
  * used to join two dataframes through pipeline
  * Don't save such pipelines.
  **/

class Joiner(override val uid: String) extends Transformer with Params {
  val right = new Param[DataFrame](this, "rDataFrame", "leftDataFrameToJoin")

  val joinType = new Param[String](this, "joinType", "joinType : inner, outer, left_outer, right_outer, leftsemi allowed",
    ParamValidators.inArray[String](Array[String]("inner", "outer", "left_outer", "right_outer", "leftsemi")))


  val joinColExpr = new Param[Column](this, "joinColExpr", "column expression to join")

  /** @group setParam */
  def setRDataFrame(value: DataFrame): this.type = set(right, value)

  /** @group getParam */
  def getLDataFrame: DataFrame = $(right)

  /** @group setParam */
  def setJoinType(value: String): this.type = set(joinType, value)

  setDefault(joinType -> "inner")

  /** @group getParam */
  def getJoinType: String = $(joinType)

  /** @group setParam */
  def setJoinColExpr(value: Column): this.type = set(joinColExpr, value)

  /** @group getParam */
  def getJoinColExpr: Column = $(joinColExpr)

  def this() = this(Identifiable.randomUID("joiner"))

  override def transform(dataset: Dataset[_]): DataFrame = {

    dataset.join($(right), $(joinColExpr), $(joinType)).toDF
  }

  override def copy(extra: ParamMap): Transformer = {
    defaultCopy(extra)
  }


  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {

    val rightDataFrame = $(right)
    val rightDFSchema = rightDataFrame.schema
    val dummyRightDataFrame = rightDataFrame.sqlContext.createDataFrame(rightDataFrame.sqlContext.sparkContext.emptyRDD[Row], rightDFSchema)
    val dummyLeftDataFrameSchema = rightDataFrame.sqlContext.createDataFrame(rightDataFrame.sqlContext.sparkContext.emptyRDD[Row], schema)

    dummyLeftDataFrameSchema.join(dummyRightDataFrame, $(joinColExpr), $(joinType)).schema
  }
}