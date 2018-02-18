package org.apache.spark.ml.odkl

import java.util.Comparator

import org.apache.spark.internal.Logging
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{ArrayType, DataType, StructType}

import scala.collection.mutable

/**
  * Created by eugeny.malyutin on 24.06.16.
  *
  * UDAF designed to extract top-numRows rows by columnValue
  * Used to replace Hive Window-functions which are to slow in case of all-df in one aggregation cell
  * Result of aggFun is packed in a column "arrData" and need to be [[org.apache.spark.sql.functions.explode]]-d
  *
  * @param numRows  num rows per aggregation colemn
  * @param dfSchema dataframe schema with all columns in one struct-column named "data"
  * @param columnToSortBy
  * @tparam B - type of columnToSortBy with implicit ordering for type B
  *
  */
class TopKUDAF[B](val numRows: Int = 20,
                  dfSchema: StructType,
                  columnToSortBy: String)
                 (implicit val cmp: Ordering[B]) extends UserDefinedAggregateFunction with Logging {

  @transient lazy val rowComparator = new Comparator[Object] {
    override def compare(o1: Object, o2: Object): Int = -cmp.compare(o1.asInstanceOf[Row].getAs[B](columnToSortByIndex), o2.asInstanceOf[Row].getAs[B](columnToSortByIndex))
  }
  val columnToSortByIndex: Int = dfSchema.fields(0).dataType.asInstanceOf[StructType].fieldIndex(columnToSortBy)

  override def bufferSchema: StructType = new StructType().add("arrData", ArrayType(dfSchema.fields.head.dataType))

  override def dataType: DataType = new StructType().add("arrData", ArrayType(dfSchema.fields.head.dataType))

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {

    var data = buffer.getAs[mutable.WrappedArray[java.lang.Object]](0).array //java.lang.Object to avoid additional copying in Seq.toArray or WrappedArray.toArray

    if (data.length < numRows) {
      val indUno = java.util.Arrays.binarySearch[Object](data, input.getAs[GenericRowWithSchema](0), rowComparator)

      val ind: (Int, Int) = if (indUno < 0) (-indUno - 1, -indUno) else (indUno, indUno + 1)
      var dataWithEl = new Array[Object](data.length + 1)

      System.arraycopy(data, 0, dataWithEl, 0, ind._1)
      dataWithEl(ind._1) = input.getAs[Object](0)

      System.arraycopy(data, ind._1, dataWithEl, ind._1 + 1, data.length - ind._1)

      data = dataWithEl
    } else {

      val currentLikes = input.getAs[GenericRowWithSchema](0).getAs[B](columnToSortByIndex)
      if (cmp.lt(data.last.asInstanceOf[Row].getAs[B](columnToSortByIndex), currentLikes)) {
        val indUno = java.util.Arrays.binarySearch[Object](data, input.getAs[GenericRowWithSchema](0), rowComparator)

        val ind = if (indUno < 0) (-indUno - 1, -indUno) else (indUno, indUno + 1)

        var dataWithEl = new Array[Row](data.length)
        System.arraycopy(data, ind._1, data, ind._1 + 1, data.length - ind._1 - 1)
        data(ind._1) = input.getAs[GenericRowWithSchema](0)

      }
    }

    buffer.update(0, data)
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    var arr1 = buffer1.getAs[mutable.WrappedArray[java.lang.Object]](0).array
    var arr2 = buffer2.getAs[mutable.WrappedArray[java.lang.Object]](0).array
    var i1 = 0
    var i2 = 0
    val ansLength = Math.min(arr1.length + arr2.length, k)
    var ans = new Array[Row](ansLength)
    var i = 0
    while (i < ansLength) {
      if (i2 >= arr2.length || i1 >= arr1.length) {
        val (ind: Int, arr: Array[Object]) = if (i2 >= arr2.length) (i1, arr1) else (i2, arr2)
        System.arraycopy(arr, ind, ans, i, Math.min(arr.length - ind, ans.length - i))
        i = ansLength
      } else if (cmp.lt(arr1(i1).asInstanceOf[Row].getAs[B](columnToSortByIndex), arr2(i2).asInstanceOf[Row].getAs[B](columnToSortByIndex))) {
        ans(i) = arr2(i2).asInstanceOf[Row]
        i2 = i2 + 1
      } else {
        ans(i) = arr1(i1).asInstanceOf[Row]
        i1 = i1 + 1
      }
      i = i + 1
    }
    buffer1.update(
      0, ans
    )
  }

  def k = numRows

  override def inputSchema: StructType = dfSchema

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = Seq.empty[Row]
  }

  override def deterministic: Boolean = true

  override def evaluate(buffer: Row): Any = buffer

}