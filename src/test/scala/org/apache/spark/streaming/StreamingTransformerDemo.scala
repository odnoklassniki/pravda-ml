package org.apache.spark.streaming

import java.util.Optional
import java.util.concurrent.atomic.{AtomicInteger, AtomicLong}
import java.util.concurrent.{ArrayBlockingQueue, TimeUnit}

import javax.annotation.concurrent.GuardedBy
import org.apache.spark.SparkEnv
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.odkl.WithTestData
import org.apache.spark.rpc.{RpcCallContext, RpcEndpointRef, RpcEnv, ThreadSafeRpcEndpoint}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.catalyst.expressions.UnsafeRow
import org.apache.spark.sql.execution.streaming.sources.{ContinuousMemoryStream, ContinuousMemoryStreamInputPartition, ContinuousMemoryStreamOffset, MemorySinkV2}
import org.apache.spark.sql.{Encoder, ForeachWriter, Row, SQLContext}
import org.apache.spark.sql.execution.streaming._
import org.apache.spark.sql.sources.v2.{ContinuousReadSupport, DataSourceOptions}
import org.apache.spark.sql.sources.v2.reader.{InputPartition, InputPartitionReader}
import org.apache.spark.sql.sources.v2.reader.streaming.{ContinuousInputPartitionReader, ContinuousReader, Offset, PartitionOffset}
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.StructType
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.JavaConverters._
import scala.collection.mutable.{ArrayBuffer, ListBuffer}


object QueueHolder {
  private val queues = new ArrayBuffer[Seq[ArrayBlockingQueue[InternalRow]]]

  def registerQueues(numPartitions : Int) : (Seq[ArrayBlockingQueue[InternalRow]], Int) = synchronized {
    val index = queues.size
    val newGroup = Seq.fill(numPartitions)(new ArrayBlockingQueue[InternalRow](1000))

    queues += newGroup

    (newGroup, index)
  }

  def getQueue(id: Int, parition: Int): ArrayBlockingQueue[InternalRow] = synchronized{queues(id)(parition)}
}

class OdklContinuousMemoryStream[A : Encoder](sqlContext: SQLContext, numPartitions: Int = 2)
  extends MemoryStreamBase[A](sqlContext) with ContinuousReader with ContinuousReadSupport {
  private implicit val formats = Serialization.formats(NoTypeHints)

  protected val logicalPlan =
    StreamingRelationV2(this, "memory", Map(), attributes, None)(sqlContext.sparkSession)

  @GuardedBy("this")
  private val (records, id) = QueueHolder.registerQueues(numPartitions)

  private val activePartition = new AtomicInteger(0)

  @GuardedBy("this")
  private var startOffset: ContinuousMemoryStreamOffset = _

  @volatile private var endpointRef: RpcEndpointRef = _

  def addData(data: TraversableOnce[A]): Offset =  {

    val partition = Math.abs(activePartition.getAndIncrement() % numPartitions)

    data.foreach(item => records(partition).add(encoder.toRow(item)))

    // The new target offset is the offset where all records in all partitions have been processed.
    ContinuousMemoryStreamOffset((0 until numPartitions).map(i => (i, records(i).size)).toMap)
  }

  override def setStartOffset(start: Optional[Offset]): Unit = synchronized {
    // Inferred initial offset is position 0 in each partition.
    startOffset = start.orElse {
      ContinuousMemoryStreamOffset((0 until numPartitions).map(i => (i, 0)).toMap)
    }.asInstanceOf[ContinuousMemoryStreamOffset]
  }

  override def getStartOffset: Offset = synchronized {
    startOffset
  }

  override def deserializeOffset(json: String): ContinuousMemoryStreamOffset = {
    ContinuousMemoryStreamOffset(Serialization.read[Map[Int, Int]](json))
  }

  override def mergeOffsets(offsets: Array[PartitionOffset]): ContinuousMemoryStreamOffset = {
    ContinuousMemoryStreamOffset(
      offsets.map {
        case ContinuousRecordPartitionOffset(part, num) => (part, num)
      }.toMap
    )
  }

  override def planInputPartitions(): java.util.List[InputPartition[InternalRow]] = {
    (0 until numPartitions).map(part =>
      new OdklContinousReader(id, part).asInstanceOf[InputPartition[InternalRow]]).toList.asJava
  }

  override def stop(): Unit = {}

  override def commit(end: Offset): Unit = {}

  // ContinuousReadSupport implementation
  // This is necessary because of how StreamTest finds the source for AddDataMemory steps.
  def createContinuousReader(
                              schema: Optional[StructType],
                              checkpointLocation: String,
                              options: DataSourceOptions): ContinuousReader = {
    this
  }
}

class OdklContinousReader(id: Int, parition: Int) extends InputPartition[InternalRow] {

  @transient private lazy val queue = QueueHolder.getQueue(id, parition)

  @transient private lazy val offset = new AtomicLong()

  override def createPartitionReader(): InputPartitionReader[InternalRow] = new ContinuousInputPartitionReader[InternalRow] {
    override def next(): Boolean = true

    override def get(): InternalRow = {
      queue.poll(1000, TimeUnit.DAYS)
    }

    override def close(): Unit = {}

    override def getOffset: PartitionOffset = ContinuousRecordPartitionOffset(parition, offset.getAndIncrement().intValue())
  }
}



case class Data(first: Double, second: Double)

class StreamingTransformerDemo extends FlatSpec with Matchers with WithTestData {

  "Streaming" should "add 2 and 2" in {


    import sqlc.implicits._

    val value: MemoryStream[Data] = MemoryStream.apply[Data](1, sqlc)

    val result = new ArrayBuffer[Double]

    val query = new SQLTransformer().setStatement("SELECT first + second AS result FROM __THIS__")
      .transform(value.toDS())
      .writeStream
      .format("memory")
      .queryName("testQuery")
      .trigger(Trigger.ProcessingTime(10))
      .start()


    val sink = query.asInstanceOf[StreamingQueryWrapper].streamingQuery.sink.asInstanceOf[MemorySink]

    value.addData(Data(2, 2))

    query.processAllAvailable()
    query.stop()
    query.awaitTermination()

    sink.allData.map(_.getDouble(0)) should contain theSameElementsAs Seq(4.0)

    query.stop()
  }

  "Streaming" should "add 2 and 2 continously" in {
    import sqlc.implicits._

    val value: MemoryStreamBase[Data] = new OdklContinuousMemoryStream[Data](sqlc, 1)

    val result = new ArrayBuffer[Double]

    val query = new SQLTransformer().setStatement("SELECT first + second AS result FROM __THIS__")
      .transform(value.toDF())
      .writeStream
      .format("memory")
      .queryName("testQuery2")
      .trigger(Trigger.Continuous(10))
      .start()


    val sink = query.asInstanceOf[StreamingQueryWrapper].streamingQuery.sink.asInstanceOf[MemorySinkV2]

    value.addData(Data(2, 2))

    query.processAllAvailable()
    query.stop()
    query.awaitTermination()

    sink.allData.map(_.getDouble(0)) should contain theSameElementsAs Seq(4.0)
  }

  "Streaming" should "batch continous partitions" in {
    import sqlc.implicits._

    val value: MemoryStreamBase[Data] = new OdklContinuousMemoryStream[Data](sqlc, 1)

    val query = value.toDS()
      .mapPartitions(i => Iterator(i.count(x => true)))
      .writeStream
      .format("memory")
      .queryName("testQuery3")
      .trigger(Trigger.Continuous(10))
      .start()

    val sink = query.asInstanceOf[StreamingQueryWrapper].streamingQuery.sink.asInstanceOf[MemorySinkV2]

    val numWrites = 30
    val writeSize = 20

    (0 until numWrites).foreach(
      x => value.addData((1 to writeSize).map(i => Data(i,i)))
    )

    query.processAllAvailable()

    Thread.sleep(1000)
    
    query.stop()
    query.awaitTermination()

    val result = sink.allData

    result.size should be <= numWrites

    result.map(_.getInt(0)).sum should be (numWrites * writeSize)

    query.stop()
  }

}
