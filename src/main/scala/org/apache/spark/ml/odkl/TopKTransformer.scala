package org.apache.spark.ml.odkl

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, explode}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, functions}

/**
  * Created by eugeny.malyutin on 20.07.17.
  *
  * Performs TopK-UDAF logic without annoying schema pack-unpack
  * @tparam B - raw type (Long for LongTyped-columns) for columnToOrderBy
  *           Ordering for this type should be defined
  */
class TopKTransformer[B](override val uid: String) (implicit val cmp: Ordering[B])
  extends Transformer with DefaultParamsWritable with HasGroupByColumns{

  val topK: IntParam =  new IntParam(this, "TopK", "number elements to find by each group",
    ParamValidators.gtEq(0))

  val columnToOrderGroupsBy: Param[String] = new Param[String](this, "columnToOrderGroupsBy",
    "column to order groups by")

  override def transform(dataset: Dataset[_]): DataFrame = {
    val guidedTempColumn = Identifiable.randomUID("tempData");

    val aggFun = new TopKUDAF[B]($(topK),  new StructType().add(guidedTempColumn,dataset.schema), $(columnToOrderGroupsBy))(cmp)
    val columnsSeq = dataset.schema.fieldNames.map(c => col(c)).toSeq
    val seqToSelect = dataset.schema.fieldNames.map(c => col("col."+c)).toSeq

    dataset
      .groupBy( $(groupByColumns).map(r => col(r)):_*)
      .agg(aggFun(functions.struct(columnsSeq: _*)).as(guidedTempColumn))
      .select(explode(col(guidedTempColumn).getField("arrData")))
      .select(seqToSelect:_*)
      .toDF
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  /** @group setParam */
  def setTopK(value: Int): this.type = set(topK, value)


  /** @group setParam */
  def setColumnToOrderGroupsBy(value: String): this.type = set(columnToOrderGroupsBy, value)

  def this()(implicit cmp:Ordering[B]) = this(Identifiable.randomUID("topKTransformer"))(cmp)

}
