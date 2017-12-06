package odkl.analysis.spark.util.collection

import scala.math.Ordering
import scala.reflect.ClassTag
import scala.util.Random

/**
 * Created by vyacheslav.baranov on 19/06/15.
 */
trait CollectionOperations {

  /**
   * Adds operations to `BufferedIterator`
   */
  implicit class ImplicitBufferedIteratorDecorator[A](it: BufferedIterator[A]) {

    /**
     * Standard `headOption`
     *
     * @return    `None` if iterator is empty, `Some(head)` otherwise
     */
    def headOption: Option[A] = if (it.hasNext) Some(it.head) else None

  }

  /**
   * Adds operations to `TraversableOnce`
   *
   * @param it
   * @tparam A
   */
  implicit class ImplicitTraversableOnceDecorator[A: ClassTag](it: TraversableOnce[A]) {

    /**
     * Selects up to `n` bottom items from `TraversableOnce`
     *
     * @param n   number of items to get
     * @param ord ordering to use
     * @return
     */
    def bottom(n: Int)(implicit ord: Ordering[A]): Seq[A] = {
      import scala.collection.convert.decorateAsScala._
      val ordTuple = implicitly[Ordering[(A, Int)]]
      val set = new java.util.TreeSet[(A, Int)](ordTuple)
      var idx = 0
      for (elem <- it) {
        val x = elem -> idx
        if (set.size < n) {
          set.add(x)
        } else if (ordTuple.gt(set.last, x)) {
          set.add(x)
          set.remove(set.last)
        }
        idx += 1
      }
      set.iterator.asScala.map(_._1).toCompactBuffer
    }

    def top(n: Int)(implicit ord: Ordering[A]): Seq[A] = bottom(n)(ord.reverse)

    def topBy[B](n: Int)(f: A => B)(implicit ord: Ordering[B]) = {
      implicit val aord = new Ordering[A] {
        override def compare(x: A, y: A): Int = ord.compare(f(x), f(y))
      }
      top(n)(aord)
    }

    def bottomBy[B](n: Int)(f: A => B)(implicit ord: Ordering[B]) = {
      val aord = new Ordering[A] {
        override def compare(x: A, y: A): Int = ord.compare(f(x), f(y))
      }
      bottom(n)(aord)
    }

    def takeSample(n: Int, rnd: Random = new Random): Seq[A] = {
      val buf = it.toBuffer
      if (buf.size <= n) buf
      else {
        for (i <- 0 until n) {
          val pos = i + rnd.nextInt(buf.size - i)
          val t = buf(i)
          buf(i) = buf(pos)
          buf(pos) = t
        }
        buf.slice(0, n)
      }
    }

    /**
     * Converts a `TraversableOnce` to `OpenHashSet` of items
     *
     * @return
     */
    def toOpenHashSet: OpenHashSet[A] = {
      val s = new OpenHashSet[A]()
      for (elem <- it) {
        s.add(elem)
      }
      s
    }

    def toCompactBuffer: CompactBuffer[A] = {
      val buf = new CompactBuffer[A]()
      for (elem <- it) {
        buf += elem
      }
      buf
    }

  }

  implicit class ImplicitArrayDecorator[A: ClassTag](a: Array[A]) {

    /**
      * Converts an `Array` to `OpenHashSet` of items.
      *
      * @return
      */
    def toOpenHashSet: OpenHashSet[A] = {
      if (a.length == 0) {
        new OpenHashSet[A](1)
      } else {
        val s = new OpenHashSet[A](a.length)
        for (elem <- a) {
          s.add(elem)
        }
        s
      }
    }

  }

  /**
   * Adds operations to `TraversableOnce[(K, V)]`
   *
   * @param it
   * @tparam K
   * @tparam V
   */
  implicit class ImplicitKVTraversableOnceDecorator[K: ClassTag, V: ClassTag](it: TraversableOnce[(K, V)]) {

    def toOpenHashMap: OpenHashMap[K, V] = {
      val m = new OpenHashMap[K, V]()
      for ((key, value) <- it) {
        m.update(key, value)
      }
      m
    }

    def groupByKey: OpenHashMap[K, CompactBuffer[V]] = {
      val m = new OpenHashMap[K, CompactBuffer[V]]()
      for ((k, v) <- it) {
        val b = m.changeValue(k, CompactBuffer[V](v), _ += v)
      }
      m
    }
  }

}

object CollectionOperations extends CollectionOperations
