package odkl.analysis.spark.util


object IteratorUtils {

  def groupByKey[Key, Value](source: Iterator[(Key, Value)]): Iterator[(Key, Iterator[Value])]
  = new GroupingIterator(source)

  def groupByKey[Key, Value](source: Iterator[Value], extractor: Value => Key): Iterator[(Key, Iterator[Value])]
  = new GroupingIterator(source.map(x => extractor(x) -> x))

  def allignByTime[Value](source: Iterator[Value], timeExtractor: Value => Long, maxDiff: Long): Iterator[(Int, Iterator[Value])]
  = new AllingingIterator(source, timeExtractor, maxDiff)

  /**
    * Simple utility to apply zero-copying grouping on pre-sorted iterator
    */
  class GroupingIterator[Key, Value](private val source: Iterator[(Key, Value)]) extends Iterator[(Key, Iterator[Value])] {

    private val buffer: BufferedIterator[(Key, Value)] = source.buffered

    private var prev: Iterator[_] = Iterator()

    def hasNext: Boolean = {
      while(prev.hasNext) prev.next()
      buffer.hasNext
    }

    def next(): (Key, Iterator[Value]) = {
      while(prev.hasNext) prev.next()

      val (firstKey, firstValue) = buffer.head


      val (prefix, suffix) =
        buffer.span(x => firstKey != null && firstKey.equals(x._1) || firstKey == null && x._1 == null)

      prev = prefix

      (firstKey, prefix.map(_._2))
    }
  }

  class AllingingIterator[Value](private val source: Iterator[Value], private val timeExtractor: Value => Long, private val maxDiff: Long) extends Iterator[(Int, Iterator[Value])] {

    private val buffer: BufferedIterator[Value] = source.buffered

    private var prev: Iterator[Value] = Iterator()
    private var index: Int = 0

    def hasNext: Boolean = {
      while(prev.hasNext) prev.next()
      buffer.hasNext
    }

    def next(): (Int, Iterator[Value]) = {
      while(prev.hasNext) prev.next()

      val firstValue = buffer.head

      var lastReturned =  timeExtractor(firstValue)

      val (pref, suff) = buffer.span(x => {
        val current = timeExtractor(x)
        if (current < lastReturned + maxDiff) {
          lastReturned = current
          true
        } else {
          false
        }
      })

      val result = (index, pref)

      index += 1
      prev = pref

      result
    }
  }
}