package org.apache.spark.ml.odkl

import java.io.{IOException, ObjectInputStream, ObjectOutputStream, OutputStream}
import java.nio.ByteBuffer

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{DefaultSerializer, Kryo, Serializer}
import com.tdunning.math.stats.AVLTreeDigest
import odkl.analysis.spark.util.Logging
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer

/**
  * Created by dmitriybugaichenko on 30.12.15.
  *
  * Utility used for estimating extended stat for the set of vectors. In addition to mean, deviation and count
  * estimates percentiles
  *
  * @param dimension   Expected dimension of vectors to aggregate
  * @param compression How should accuracy be traded for size?  A value of N here will give quantile errors
  *                    almost always less than 3/N with considerably smaller errors expected for extreme
  *                    quantiles. Conversely, you should expect to track about 5 N centroids for this
  *                    accuracy.
  */
class ExtendedMultivariateOnlineSummarizer
(
  val dimension: Int,
  val compression: Double) extends MultivariateOnlineSummarizer with Serializable with Logging {

  val percentileAggregators = Array.tabulate(dimension) { i => new SeriallizableAvlTreeDigest(compression) }

  override def add(sample: mllib.linalg.Vector): this.type = {
    require(sample.size == dimension, s"Expecting vector of size $dimension")
    super.add(sample)

    for (i <- 0 until dimension) percentileAggregators(i).add(sample(i))

    this
  }

  override def merge(other: MultivariateOnlineSummarizer): this.type = {
    require(other.isInstanceOf[ExtendedMultivariateOnlineSummarizer], "Extended summarizer expected.")
    require(dimension == other.asInstanceOf[ExtendedMultivariateOnlineSummarizer].dimension, s"Expecting summarizer with the same dimansion $dimension")

    super.merge(other)

    for (i <- 0 until dimension) try {
      percentileAggregators(i).add(other.asInstanceOf[ExtendedMultivariateOnlineSummarizer].percentileAggregators(i))
    } catch {
      case e: Exception => logError(s"Exception while aggregating index $i", e)
    }

    this
  }

  def percentile(p: Double): Vector = {
    require(p > 0 && p < 1, "Expected p between 0 and 1 both excluding")
    require(count > 0, "Expected at least one sample")

    Vectors.dense(Array.tabulate(dimension) {
      i => percentileAggregators(i).percentile(p)
    })
  }
}

/**
  * Serializable wrapper over the TDigest
  *
  * @param initialCompression How should accuracy be traded for size?  A value of N here will give quantile errors
  *                           almost always less than 3/N with considerably smaller errors expected for extreme
  *                           quantiles.  Conversely, you should expect to track about 5 N centroids for this
  *                           accuracy.
  */
@DefaultSerializer(classOf[SeriallizableAvlTreeDigest])
class SeriallizableAvlTreeDigest(val initialCompression: Double = 100) extends Serializer[SeriallizableAvlTreeDigest] with Serializable {
  def this() = this(100)

  def this(dig : AVLTreeDigest) = {
    this(dig.compression())
    digest = dig
  }

  var digest = new AVLTreeDigest(initialCompression)

  def effectiveCompression = digest.compression()

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    writeToOutputStream(out)
  }

  def writeToOutputStream(out: OutputStream): Unit = {
    val bytes = Array.ofDim[Byte](digest.smallByteSize() + 4)
    val buffer: ByteBuffer = ByteBuffer.wrap(bytes)

    digest.asSmallBytes(buffer.putInt(bytes.length - 4))
    out.write(bytes)
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    val size: Int = in.readInt()
    val bytes = Array.ofDim[Byte](size)
    in.readFully(bytes)

    digest = AVLTreeDigest.fromBytes(ByteBuffer.wrap(bytes))
  }

  def add(x: Double) = digest.add(x)

  def add(x: SeriallizableAvlTreeDigest) = digest.add(x.digest)

  def percentile(p: Double) = {
    require(p > 0 && p < 1, "Expected p between 0 and 1 both excluding")
    digest.quantile(p)
  }

  override def write(kryo: Kryo, output: Output, obj: SeriallizableAvlTreeDigest): Unit = {
    obj.writeToOutputStream(output)
  }

  override def read(kryo: Kryo, input: Input, `type`: Class[SeriallizableAvlTreeDigest]): SeriallizableAvlTreeDigest = {
    val size = input.readInt()
    val bytes = Array.ofDim[Byte](size)
    input.read(bytes)

    new SeriallizableAvlTreeDigest(AVLTreeDigest.fromBytes(ByteBuffer.wrap(bytes)))
  }
}