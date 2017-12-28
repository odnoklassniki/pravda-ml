package org.apache.spark.ml.odkl

import java.util.Random

import odkl.analysis.spark.TestEnv
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.sql.{Column, Row, functions}

/**
  * Created by dmitriybugaichenko on 24.01.16.
  */
trait WithTestData extends TestEnv {

  val delta: Double = 0.001
  val hiddenModel = Vectors.dense(0.5, -0.2)
  val hiddenIntercept = 0.2

  case class Instance(label: Double, first: Double, second: Double)

  val logistic = functions.udf[Double, Double](x => {
    WithTestData.logit(x)
  })

  def cosineDistance(v1: Vector, v2: Vector): Double = {
    1.0 - BLAS.dot(v1, v2) / Math.sqrt(BLAS.dot(v1, v1) * BLAS.dot(v2, v2))
  }

  lazy val noInterceptData = WithTestData._noInterceptData
  lazy val interceptData = WithTestData._interceptData

  lazy val noInterceptDataLogistic = WithTestData._noInterceptDataLogistic
  lazy val interceptDataLogistig = WithTestData._interceptDataLogistig

  val hash = functions.udf[Int, Row](x => x.hashCode)


  val typeAssign = functions.udf[String, Int](x => if (x >= 0) "Direct" else "Inverse")
  val invertLabel = functions.udf[Double, String, Double]((t, l) => if ("Direct".equalsIgnoreCase(t)) l else 1 - l)
  val assignClass = functions.udf[String, Double](x => if (x > 0.0) "Positive" else "Negative")

  lazy val typedWithLabels = WithTestData._typedWithLabels
  lazy val withClass = WithTestData._withClass
  lazy val withTypeAndClass = WithTestData._withTypeAndClass

  lazy val rawData = WithTestData._rawData
}

object WithTestData extends TestEnv with WithTestData {

  val random = new Random(0xdeadbeaf)

  def logit(x: Double): Double = {
    val prob: Double = 1.0 / (1.0 + Math.exp(-x))

    if (prob >= 0.5) 1.0 else 0.0
  }

  @transient lazy val _rawData = sqlc.createDataFrame(
    Array.tabulate(1000) { i => {
      val x = Vectors.dense(random.nextDouble() * 2 - 1.0, random.nextDouble() * 2 - 1.0)
      Instance(label = BLAS.dot(x, hiddenModel), first = x(0), second = x(1))
    }
    })

  @transient lazy val _noInterceptData = new VectorAssembler()
    .setInputCols(Array("first", "second"))
    .setOutputCol("features").transform(_rawData)

  @transient lazy val _interceptData = _noInterceptData.withColumn("label", _noInterceptData("label") + functions.lit(hiddenIntercept))

  @transient lazy val _noInterceptDataLogistic = _noInterceptData.withColumn("label", logistic(_noInterceptData("label")))
  @transient lazy val _interceptDataLogistig = _interceptData.withColumn("label", logistic(_interceptData("label")))

  @transient lazy val columns: Seq[Column] = noInterceptDataLogistic.schema.fields.map(f => noInterceptDataLogistic(f.name)).toSeq
  @transient lazy val withHash = noInterceptDataLogistic.withColumn("hash", hash(functions.struct(columns: _*)))

  @transient lazy val typedData = withHash.withColumn("type", typeAssign(withHash("hash")))
  @transient lazy val _typedWithLabels = typedData.withColumn("label", invertLabel(typedData("type"), typedData("label")))
  @transient lazy val _withClass = noInterceptDataLogistic.withColumn("classes", assignClass(noInterceptDataLogistic("label"))).drop("label")
  @transient lazy val _withTypeAndClass = typedWithLabels.withColumn("classes", assignClass(typedWithLabels("label"))).drop("label")
}
