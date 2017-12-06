package odkl.analysis.spark.util

import odkl.analysis.spark.util.collection.CompactBuffer

import scala.reflect.ClassTag

/**
  * Created by dmitriybugaichenko on 17.05.17.
  *
  * Simple utility to apply zero-copying grouping on pre-sorted iterator
  */
class GroupingIterator[Key: ClassTag, Value: ClassTag](private val source: Iterator[(Key, Value)]) extends Iterator[(Key, Seq[Value])] {

  private var currentKey: Key = _
  private var currentValue: Value = _
  private var currentInitialized : Boolean = false

  def hasNext: Boolean = source.hasNext || currentInitialized

  def next(): (Key, Seq[Value]) = {
    if (!currentInitialized) {
      val (firstKey, firstValue) = source.next()
      currentKey = firstKey
      currentValue = firstValue
      currentInitialized = true
    }

    val keyToReturn = currentKey
    val buffer = CompactBuffer[Value](currentValue)

    while(source.hasNext) {
      val (key, value) = source.next()
      if (currentKey != null && currentKey.equals(key) || currentKey == null && key == null) {
        buffer += value
      } else {
        currentKey = key
        currentValue = value

        return (keyToReturn, buffer)
      }
    }

    currentInitialized = false
    (keyToReturn, buffer)
  }
}

object GroupingIterator {
  def apply[Key: ClassTag,Value: ClassTag](source: Iterator[(Key, Value)]) : Iterator[(Key,Seq[Value])]
    = new GroupingIterator(source)

  def apply[Key: ClassTag,Value: ClassTag](source: Iterator[Value], extractor: Value => Key) : Iterator[(Key,Seq[Value])]
    = new GroupingIterator(source.map(x => extractor(x) -> x))
}
