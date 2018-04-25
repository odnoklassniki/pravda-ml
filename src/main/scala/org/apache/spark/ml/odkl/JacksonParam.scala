package org.apache.spark.ml.odkl

import com.fasterxml.jackson.core.{JsonGenerator, JsonParser}
import com.fasterxml.jackson.databind.deser.std.PrimitiveArrayDeserializers
import com.fasterxml.jackson.databind.module.SimpleModule
import com.fasterxml.jackson.databind._
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.DenseVector

import scala.reflect.ClassTag

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utility for serializing complex parameters using jackson (handles few
  * types automatically which can not be handled by json4s)
  */


class JacksonParam[T] (
                        parent: String,
                        name: String,
                        doc: String,
                        isValid: T => Boolean,
                        default: Option[T]
                      )(implicit ct: ClassTag[T])
  extends Param[T](parent, name, doc, isValid) with Logging {

  def this(parent: Identifiable, name: String, doc: String, isValid: T => Boolean)(implicit ct: ClassTag[T]) = {
    this(parent.uid, name, doc, isValid, None)
  }

  def this(parent: String, name: String, doc: String)(implicit ct: ClassTag[T]) = {
    this(parent, name, doc, (x: T) => true, None)
  }

  def this(parent: Identifiable, name: String, doc: String)(implicit ct: ClassTag[T]) = {
    this(parent.uid, name, doc)
  }

  override def jsonEncode(value: T): String = {
    JacksonParam.objectMapper.writeValueAsString(value)
  }

  override def jsonDecode(json: String): T = {
    try {
      JacksonParam.objectMapper.readValue[T](json, ct.runtimeClass.asInstanceOf[Class[T]])
    } catch {
      case e: Throwable =>
        logError(s"Failed to read param $name from data $json due error", e)
        default.get
    }
  }

}

object JacksonParam extends Serializable {

  def apply[T](parent: Identifiable, name: String, doc: String,
               isValid: (T) => Boolean = (x: T) => true,
               default: Option[T] = None)(implicit ct: ClassTag[T]) = {
    new JacksonParam[T](parent.uid, name, doc, isValid, default)
  }

  def mapParam[V](parent: Identifiable, name: String, doc: String,
                  isValid: (Map[String, V]) => Boolean = (x: Map[String, V]) => true,
                  default: Option[Map[String, V]] = Some(Map[String,V]()))(implicit ct: ClassTag[Map[String, V]]) = {
    new JacksonParam[Map[String, V]](parent.uid, name, doc, isValid, default)
  }

  def arrayParam[V](parent: Identifiable, name: String, doc: String)(implicit ct: ClassTag[Array[V]], cv: ClassTag[V]) = {
    new JacksonParam[Array[V]](parent.uid, name, doc, (x : Array[V]) => true, Some[Array[V]](Array[V]()))
  }

  lazy val objectMapper: ObjectMapper = createDefaultMapper

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

}

