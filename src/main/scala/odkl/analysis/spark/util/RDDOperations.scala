package odkl.analysis.spark.util

import odkl.analysis.spark.util.collection.CollectionOperations
import org.apache.spark.Partitioner
import org.apache.spark.odkl.SparkUtils
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Created by vyacheslav.baranov on 19/10/15.
 */
trait RDDOperations {

  implicit class PairRDDDecorator[K: ClassTag, A: ClassTag](rdd: RDD[(K, A)]) {

    /**
     * Creates an RDD having each of required objects distributed to all required destinations
     *
     * If the `data` and/or `mapping` are partitioned with the same partitioner, no shuffle is required.
     *
     * @param mapping       Mapping of keys to destinations. Duplicate destinations from different partitions are coalesced.
     * @param partitioner   Partitioner to use. Operation is more efficient if data is partitioned with the same partitioner
     * @param f             Function to apply to values. It's called exactly once for each value in `data`
     * @tparam B
     * @return
     */
    def distributeToPartitions[B, D: ClassTag](mapping: RDD[(K, D)], partitioner: Partitioner)
                               (f: A => B): RDD[(D, (K, B))] = {
      RDDOperations.distributeToPartitions(rdd, mapping, partitioner)(f)
    }

    /**
     * Similar to [[org.apache.spark.rdd.PairRDDFunctions.join]], but ensures that all keys are unique in both RDDs.
     *
     * Throws [[java.lang.IllegalArgumentException]] if encounters duplicate
     *
     * @param other
     * @tparam B
     * @return
     */
    def joinUnique[B: ClassTag](other: RDD[(K, B)]): RDD[(K, (A, B))] = {
      RDDOperations.joinUnique(rdd, other)
    }

    /**
     * Similar to [[org.apache.spark.rdd.PairRDDFunctions.join]], but ensures that all keys are unique in both RDDs.
     *
     * Throws [[java.lang.IllegalArgumentException]] if encounters duplicate
     *
     * @param other
     * @tparam B
     * @return
     */
    def joinUnique[B: ClassTag](other: RDD[(K, B)], numPartitions: Int): RDD[(K, (A, B))] = {
      RDDOperations.joinUnique(rdd, other, numPartitions)
    }

    /**
     * Similar to [[org.apache.spark.rdd.PairRDDFunctions.join]], but ensures that all keys are unique in both RDDs.
     *
     * Throws [[java.lang.IllegalArgumentException]] if encounters duplicate
     *
     * @param other
     * @tparam B
     * @return
     */
    def joinUnique[B: ClassTag](other: RDD[(K, B)], partitioner: Partitioner): RDD[(K, (A, B))] = {
      RDDOperations.joinUnique(rdd, other, partitioner)
    }


    /**
     * Similar to [[org.apache.spark.rdd.PairRDDFunctions.leftOuterJoin]], but ensures that all keys are unique in both RDDs.
     *
     * Throws [[java.lang.IllegalArgumentException]] if encounters duplicate
     *
     * @param other
     * @tparam B
     * @return
     */
    def leftJoinUnique[B: ClassTag](other: RDD[(K, B)]): RDD[(K, (A, Option[B]))] = {
      RDDOperations.leftJoinUnique(rdd, other)
    }

    /**
     * Similar to [[org.apache.spark.rdd.PairRDDFunctions.leftOuterJoin]], but ensures that all keys are unique in both RDDs.
     *
     * Throws [[java.lang.IllegalArgumentException]] if encounters duplicate
     *
     * @param other
     * @tparam B
     * @return
     */
    def leftJoinUnique[B: ClassTag](other: RDD[(K, B)], numPartitions: Int): RDD[(K, (A, Option[B]))] = {
      RDDOperations.leftJoinUnique(rdd, other, numPartitions)
    }

    /**
     * Similar to [[org.apache.spark.rdd.PairRDDFunctions.leftOuterJoin]], but ensures that all keys are unique in both RDDs.
     *
     * Throws [[java.lang.IllegalArgumentException]] if encounters duplicate
     *
     * @param other
     * @tparam B
     * @return
     */
    def leftJoinUnique[B: ClassTag](other: RDD[(K, B)], partitioner: Partitioner): RDD[(K, (A, Option[B]))] = {
      RDDOperations.leftJoinUnique(rdd, other, partitioner)
    }

    /**
      * Utility used to group data by key within RDD partitions by a key, assuming that RDD is already partitioned
      * and sorted by key. The RDD data are processed sequentialy without shuffling and materializing them in memory.
      * @return RDD with key -> values.
      */
    def groupWithinPartitionsByKey: RDD[(K,Iterator[A])] = {
      RDDOperations.groupWithinPartitions(rdd)
    }
  }

  implicit class ImplicitRDDDecorator[T: ClassTag](rdd: RDD[T]) extends Serializable {

    /**
     * Maps RDD preserving partitioning
     *
     * @param f
     * @tparam B
     * @return
     */
    def mapPreserve[B: ClassTag](f: T => B): RDD[B] = SparkUtils.withScope(rdd.sparkContext, allowNesting = false) {
      rdd.mapPartitions(_.map(f), preservesPartitioning = true)
    }

    /**
      * Utility used to group data by key within RDD partitions by a key, assuming that RDD is already partitioned
      * and sorted by key. The RDD data are processed sequentialy without shuffling and materializing them in memory.
      * @param expr Expression to extract key from a value.
      * @return RDD with key -> values.
      */
    def groupWithinPartitionsBy[K: ClassTag](expr: T => K): RDD[(K,Iterator[T])] = {
      RDDOperations.groupWithinPartitions(rdd.map(x => expr(x) -> x))
    }
  }

}

object RDDOperations extends RDDOperations with CollectionOperations with Serializable {

  /**
   * Creates an RDD having each of required objects distributed to all required destinations
   *
   * If the `data` and/or `mapping` are partitioned with the same partitioner, no shuffle is required. 
   *
   * @param data          Source data to process
   * @param mapping       Mapping of keys to destinations. Duplicate destinations from different partitions are coalesced.
   * @param partitioner   Partitioner to use. Operation is more efficient if data is partitioned with the same partitioner
   * @param f             Function to apply to values. It's called exactly once for each value in `data`
   * @tparam K
   * @tparam A
   * @tparam B
   * @return
   */
  def distributeToPartitions[K: ClassTag, A: ClassTag, B, D: ClassTag](data: RDD[(K, A)], mapping: RDD[(K, D)], partitioner: Partitioner)
                                   (f: A => B): RDD[(D, (K, B))] = SparkUtils.withScope(data.sparkContext, allowNesting = false) {
    val res = data.cogroup(mapping, partitioner).flatMap { case (key, (it1, it2)) =>
      if (it1.isEmpty || it2.isEmpty) {
        Iterator()
      } else {
        val seq = it1.toSeq
        require(seq.size == 1, s"Multiple objects exist for key $key: $seq")

        //Coalesce destinations
        val destinations = it2.toOpenHashSet

        //Produce result
        val r = f(seq.head)
        for (p <- destinations.iterator) yield p -> (key, r)
      }
    }
    res
  }

  def joinUnique[K: ClassTag, A: ClassTag, B: ClassTag](
      rdd: RDD[(K, A)],
      other: RDD[(K, B)]): RDD[(K, (A, B))] = SparkUtils.withScope(rdd.sparkContext, allowNesting = false) {

    rdd.cogroup(other)
      .flatMap(RDDOperations.uniqueJoinedMapper[K, A, B])
  }

  def joinUnique[K: ClassTag, A: ClassTag, B: ClassTag](
    rdd: RDD[(K, A)],
      other: RDD[(K, B)],
      numPartitions: Int): RDD[(K, (A, B))] = SparkUtils.withScope(rdd.sparkContext, allowNesting = false) {

    rdd.cogroup(other)
      .flatMap(RDDOperations.uniqueJoinedMapper[K, A, B])
  }

  def joinUnique[K: ClassTag, A: ClassTag, B: ClassTag](
      rdd: RDD[(K, A)],
      other: RDD[(K, B)],
      partitioner: Partitioner): RDD[(K, (A, B))] = SparkUtils.withScope(rdd.sparkContext, allowNesting = false) {

    rdd.cogroup(other)
      .flatMap(RDDOperations.uniqueJoinedMapper[K, A, B])
  }

  def leftJoinUnique[K: ClassTag, A: ClassTag, B: ClassTag](
    rdd: RDD[(K, A)],
    other: RDD[(K, B)]): RDD[(K, (A, Option[B]))] = SparkUtils.withScope(rdd.sparkContext, allowNesting = false) {

    rdd.cogroup(other)
      .flatMap(RDDOperations.uniqueLeftJoinedMapper[K, A, B])
  }

  def leftJoinUnique[K: ClassTag, A: ClassTag, B: ClassTag](
    rdd: RDD[(K, A)],
    other: RDD[(K, B)],
    numPartitions: Int): RDD[(K, (A, Option[B]))] = SparkUtils.withScope(rdd.sparkContext, allowNesting = false) {

    rdd.cogroup(other)
      .flatMap(RDDOperations.uniqueLeftJoinedMapper[K, A, B])
  }

  def leftJoinUnique[K: ClassTag, A: ClassTag, B: ClassTag](
    rdd: RDD[(K, A)],
    other: RDD[(K, B)],
    partitioner: Partitioner): RDD[(K, (A, Option[B]))] = SparkUtils.withScope(rdd.sparkContext, allowNesting = false) {

    rdd.cogroup(other)
      .flatMap(RDDOperations.uniqueLeftJoinedMapper[K, A, B])
  }

  /**
    * Utility used to group data by key within RDD partitions by a key, assuming that RDD is already partitioned
    * and sorted by key. The RDD data are processed sequentialy without shuffling and materializing them in memory.
    * @param rdd RDD to group
    * @tparam K Type of the key
    * @tparam V Type of the value
    * @return RDD with key -> values.
    */
  def groupWithinPartitions[K: ClassTag, V: ClassTag](rdd: RDD[(K,V)]): RDD[(K,Iterator[V])] = {
    rdd.mapPartitions(x => IteratorUtils.groupByKey(x))
  }

  private def uniqueJoinedMapper[K: ClassTag, A: ClassTag, B: ClassTag](
      v: (K, (Iterable[A], Iterable[B]))): Iterator[(K, (A, B))] = {
    val key = v._1
    val iter1 = v._2._1
    val iter2 = v._2._2
    require(iter1.size <= 1, s"Expected at most 1 item for key '$key' in left RDD, actually: ${iter1.size}")
    require(iter2.size <= 1, s"Expected at most 1 item for key '$key' in right RDD, actually: ${iter2.size}")
    if (iter1.size == 1 && iter2.size == 1) {
      Iterator(key -> (iter1.head, iter2.head))
    } else {
      Iterator()
    }

  }

  private def uniqueLeftJoinedMapper[K: ClassTag, A: ClassTag, B: ClassTag](
    v: (K, (Iterable[A], Iterable[B]))): Iterator[(K, (A, Option[B]))] = {
    val key = v._1
    val iter1 = v._2._1
    val iter2 = v._2._2
    require(iter1.size <= 1, s"Expected at most 1 item for key '$key' in left RDD, actually: ${iter1.size}")
    require(iter2.size <= 1, s"Expected at most 1 item for key '$key' in right RDD, actually: ${iter2.size}")
    if (iter1.size == 1) {
      Iterator(key -> (iter1.head, iter2.headOption))
    } else {
      Iterator()
    }
  }
}
