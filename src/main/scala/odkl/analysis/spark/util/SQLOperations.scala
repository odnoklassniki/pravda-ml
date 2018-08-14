package odkl.analysis.spark.util

import odkl.analysis.spark.util.collection.CompactBuffer
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.exception.util.LocalizedFormats
import org.apache.commons.math3.exception.{NotPositiveException, NotStrictlyPositiveException, NumberIsTooLargeException, OutOfRangeException}
import org.apache.commons.math3.util.FastMath
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction, UserDefinedFunction}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}

import scala.annotation.tailrec
import scala.reflect.ClassTag

/**
 * Created by vyacheslav.baranov on 25/11/15.
 */
trait SQLOperations {

  def collectAsList(dataType: DataType): UserDefinedAggregateFunction = dataType match {
    case IntegerType => new SQLOperations.CollectAsList[Int](dataType)
    case LongType => new SQLOperations.CollectAsList[Long](dataType)
    case FloatType => new SQLOperations.CollectAsList[Float](dataType)
    case DoubleType => new SQLOperations.CollectAsList[Double](dataType)
    case StringType => new SQLOperations.CollectAsList[String](dataType)
    case _ => new SQLOperations.CollectAsList[Any](dataType)
  }

  def collectAsSet(dataType: DataType): UserDefinedAggregateFunction = dataType match {
    case IntegerType => new SQLOperations.CollectAsSet[Int](dataType)
    case LongType => new SQLOperations.CollectAsSet[Long](dataType)
    case FloatType => new SQLOperations.CollectAsSet[Float](dataType)
    case DoubleType => new SQLOperations.CollectAsSet[Double](dataType)
    case StringType => new SQLOperations.CollectAsSet[String](dataType)
    case _ => throw new UnsupportedOperationException("Collect set supported only for int, long, float, double and string")
  }

  def mergeSets(dataType: DataType): UserDefinedAggregateFunction = dataType match {
    case IntegerType => new SQLOperations.MergeSets[Int](dataType)
    case LongType => new SQLOperations.MergeSets[Long](dataType)
    case FloatType => new SQLOperations.MergeSets[Float](dataType)
    case DoubleType => new SQLOperations.MergeSets[Double](dataType)
    case StringType => new SQLOperations.MergeSets[String](dataType)
    case _ => throw new UnsupportedOperationException("Merge sets supported only for int, long, float, double and string")
  }

  /**
    * Utility used to create UDF-s for Willson confidence interval estimation. Mainly copied from
    * org.apache.commons.math3.stat.interval.WilsonScoreInterval, but optimized for multiple evaluations.
    *
    * See https://habrahabr.ru/company/darudar/blog/143188/ for details.
    *
    * @param confidence Confidence level (95% by default)
    * @param minBound Minimum lower bound value (bellow that result is reset to 0 to achive higher sparsity).
    * @return Function for estimation of Wilson confidence interval lower bound.
    */
  def willsonLower(sqlContext: SQLContext, confidence: Double = 0.95, minBound: Double = 0.0): UserDefinedFunction = {
    if (confidence <= 0 || confidence >= 1)
      throw new OutOfRangeException(LocalizedFormats.OUT_OF_BOUNDS_CONFIDENCE_LEVEL, confidence, 0, 1)

    val alpha = (1.0 - confidence) / 2
    val normalDistribution = new NormalDistribution
    val z = normalDistribution.inverseCumulativeProbability(1 - alpha)
    val zSquared = FastMath.pow(z, 2)

    sqlContext.udf.register(
      s"wilsonLowerBound_${Math.round(confidence * 100)}",
      (positive: Long, total: Long) => {
        if (total <= 0) throw new NotStrictlyPositiveException(LocalizedFormats.NUMBER_OF_TRIALS, total)
        if (positive < 0) throw new NotPositiveException(LocalizedFormats.NEGATIVE_NUMBER_OF_SUCCESSES, positive)
        if (positive > total) throw new NumberIsTooLargeException(LocalizedFormats.NUMBER_OF_SUCCESS_LARGER_THAN_POPULATION_SIZE, positive, total, true)

        val result = if (total <= 0) 0.0 else {
          val mean = positive.toDouble / total.toDouble
          val factor = 1.0 / (1 + (1.0 / total) * zSquared)
          val modifiedSuccessRatio = mean + (1.0 / (2 * total)) * zSquared
          val difference = z * FastMath.sqrt(1.0 / total * mean * (1 - mean) + (1.0 / (4 * FastMath.pow(total, 2)) * zSquared))

          factor * (modifiedSuccessRatio - difference)
        }

        if (result < minBound) 0 else result
      })
  }

}

object SQLOperations extends SQLOperations {

  class CollectAsList[T: ClassTag](itemType: DataType) extends UserDefinedAggregateFunction {
    override def inputSchema: StructType = new StructType()
      .add("item", itemType)

    override def bufferSchema: StructType = new StructType()
      .add("items", new ArrayType(itemType, true))

    override def dataType: DataType = new ArrayType(itemType, true)

    override def deterministic: Boolean = true

    override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
      val prevItems = buffer.getAs[Seq[T]](0)
      val res = appendItem(prevItems, input.getAs[T](0))
      buffer.update(0, res)
    }

    override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
      val prevItems1 = buffer1.getAs[Seq[T]](0)
      val prevItems2 = buffer2.getAs[Seq[T]](0)
      val res = appendItems(prevItems1, prevItems2)
      buffer1.update(0, res)
    }

    override def initialize(buffer: MutableAggregationBuffer): Unit = {
      buffer.update(0, new CompactBuffer[T]())
    }

    override def evaluate(buffer: Row): Any = {
      val items = buffer.getAs[Seq[T]](0)
      items
    }

    protected def appendItem(buf: Seq[T], item: T): Seq[T] = buf match {
      case cbuf: CompactBuffer[T] =>
        cbuf += item
        cbuf
      case _ =>
        val res = new CompactBuffer[T](buf.length + 1)
        res ++= buf
        res += item
        res
    }

    protected def appendItems(buf: Seq[T], items: Seq[T]): Seq[T] = buf match {
      case cbuf: CompactBuffer[T] =>
        cbuf ++= items
        cbuf
      case _ =>
        val res = new CompactBuffer[T](buf.length + items.length)
        res ++= buf
        res ++= items
        res
    }

  }

  class CollectAsSet[T: ClassTag](itemType: DataType)(implicit ordering: Ordering[T]) extends CollectAsList[T](itemType) {

    override protected def appendItem(buf: Seq[T], item: T): Seq[T] =  {
      if (item == null) {
        buf
      } else {
        val array: Array[T] = buf.toArray

        if (binarySearch(array, item) < 0) {
          Array.concat(array, Array(item)).sorted
        } else {
          array
        }
      }
    }

    override protected  def appendItems(buf: Seq[T], items: Seq[T]): Seq[T] = {
      val array = buf.toArray

      val newItems = items.filter(x => x != null && binarySearch(array, x) < 0).toArray

      val result = if (newItems.isEmpty) {
        array
      } else {
        Array.concat(array, newItems).sorted
      }

      result
    }

    def binarySearch(a: IndexedSeq[T], needle: T): Int = {
      @tailrec
      def binarySearch(low: Int, high: Int): Int = {
        if (low <= high) {
          val middle = low + (high - low) / 2

          if (ordering.equiv(a(middle), needle))
            middle
          else if (ordering.lt(a(middle), needle))
            binarySearch(middle + 1, high)
          else
            binarySearch(low, middle - 1)
        } else
          -(low + 1)
      }

      binarySearch(0, a.length - 1)
    }
  }

  class MergeSets[T: ClassTag](itemType: DataType)(implicit ordering: Ordering[T]) extends CollectAsSet[T](itemType) {

    override def inputSchema: StructType = new StructType()
      .add("item", ArrayType(itemType))


    override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
      val prevItems = buffer.getAs[Seq[T]](0)
      val newItems = input.getAs[Seq[T]](0)
      if (newItems != null && newItems.nonEmpty) {
        val res = appendItems(prevItems, newItems)
        buffer.update(0, res)
      }
    }
  }

  /**
   * Defined whether to choose the last collated event before the item of the first event after the item.
   */
  object CollateOrder extends Enumeration {
    val Before, After = Value
  }


  /**
   * Returns a key extractor
   *
   * Currently, optimized for keys of 1 or 2 fields.
   *
   * @param indexes
   * @return
   */
  private def keyExtractor(indexes: Seq[Int]): (Row => Any) = {
    if (indexes.length == 1) new SingleKeyExtractor(indexes.head)
    else if (indexes.length == 2) new Tuple2Extractor(indexes.head, indexes(1))
    else new SeqKeyExtractor(indexes)
  }

  private class SingleKeyExtractor(i: Int) extends (Row => Any) with Serializable {
    override def apply(row: Row): Any = row.get(i)
  }

  private class Tuple2Extractor(i1: Int, i2: Int) extends (Row => Any) with Serializable {
    override def apply(row: Row): Any = row.get(i1) -> row.get(i2)
  }

  private class SeqKeyExtractor(indexes: Seq[Int]) extends (Row => Any) with Serializable {
    override def apply(row: Row): Any = indexes.map(row.get)
  }
  

}
