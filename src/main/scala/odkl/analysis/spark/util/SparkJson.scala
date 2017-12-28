package odkl.analysis.spark.util

import java.io.{ByteArrayInputStream, InputStream}
import java.net.URI

import com.fasterxml.jackson.core.`type`.TypeReference
import com.fasterxml.jackson.core.{JsonGenerator, JsonParser}
import com.fasterxml.jackson.databind._
import com.fasterxml.jackson.databind.deser.std.PrimitiveArrayDeserializers
import com.fasterxml.jackson.databind.module.SimpleModule
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.mllib.linalg.DenseVector

/**
 * Collection of JSON utility functions for Spark jobs
 *
 * Created by vyacheslav.baranov on 20/10/15.
 */
trait SparkJson {

  def objectMapper = SparkJson._objectMapper

  implicit class ImplicitObjectMapperDecorator(mapper: ObjectMapper) {

    def writeValue(conf: Configuration, location: String, value: Any, overwrite: Boolean = false): Unit = {
      val uri = new URI(location)
      val fs = FileSystem.get(uri, conf)
      val path = new Path(location)
      if (fs.exists(path) && overwrite) {
        fs.delete(path, true)
      }
      if (fs.exists(path)) {
        throw new RuntimeException(s"File $location already exists, use overwrite flag to force overwrite")
      }
      val os = fs.create(path)
      try {
        mapper.writeValue(os, this)
        os.flush()
      } finally {
        os.close()
      }
    }

    def readValue[T](conf: Configuration, location: String, valueClass: Class[T]) = {
      val uri = new URI(location)
      val fs = FileSystem.get(uri, conf)
      val path = new Path(location)
      val is = fs.open(path)
      try {
        mapper.readValue(is, valueClass)
      } finally {
        is.close()
      }
    }

    def readValue[T](conf: Configuration, location: String, valueTypeRef: TypeReference[T]) = {
      val uri = new URI(location)
      val fs = FileSystem.get(uri, conf)
      val path = new Path(location)
      val is = fs.open(path)
      try {
        mapper.readValue(is, valueTypeRef)
      } finally {
        is.close()
      }
    }

  }
}

object SparkJson extends SparkJson {

  private lazy val _objectMapper: ObjectMapper = createDefaultMapper

  private def createDefaultMapper: ObjectMapper = {
    val m = new ObjectMapper()
    m.registerModule(DefaultScalaModule)
    m.registerModule(sparkModule)
    m
  }

  def sparkModule: Module = {
    val module = new SimpleModule("SparkJson")
    module.addSerializer(classOf[DenseVector], DenseVectorSerializer)
    module.addDeserializer(classOf[DenseVector], DenseVectorDeserializer)


    module
  }

}

private object DenseVectorSerializer extends JsonSerializer[DenseVector] {

  override def serialize(value: DenseVector, gen: JsonGenerator, serializers: SerializerProvider): Unit = {
    gen.writeStartArray(value.size)
    for (i <- 0 until value.size) gen.writeNumber(value(i))
    gen.writeEndArray()
  }
}

private object DenseVectorDeserializer extends JsonDeserializer[DenseVector] {

  private val arrayDeserializer = PrimitiveArrayDeserializers.forType(java.lang.Double.TYPE)

  override def deserialize(p: JsonParser, ctxt: DeserializationContext): DenseVector = {
    val array = arrayDeserializer.deserialize(p, ctxt).asInstanceOf[Array[Double]]
    new DenseVector(array)
  }
}