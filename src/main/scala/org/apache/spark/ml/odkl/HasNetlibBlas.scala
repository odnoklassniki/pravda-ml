package org.apache.spark.ml.odkl

import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}
import com.github.fommil.netlib.{F2jBLAS, BLAS => NetlibBLAS}
import org.apache.spark.ml.linalg.{DenseVector, Matrices, Vector, Vectors}
/**
  * Created by dmitriybugaichenko on 19.11.16.
  *
  * Utility for simplifying BLAS access.
  */
trait HasNetlibBlas {
  // For level-1 routines, we use Java implementation.
  def f2jBLAS: NetlibBLAS = HasNetlibBlas._f2jBLAS

  def blas: NetlibBLAS = HasNetlibBlas._nativeBLAS

  def dscal(a: Double, data: Array[Double]) : Unit = f2jBLAS.dscal(data.length, a, data, 1)

  def axpy(a: Double, x: Array[Double], y : Array[Double]) : Unit = f2jBLAS.daxpy(x.length, a, x, 1, y, 1)

  def axpy(a: Double, x: Vector, y : Array[Double]) : Unit = x match {
    case dense: DenseVector => axpy(a, dense.values, y)
    case _ => x.foreachActive((i, v) => y(i) += a * v)
  }

  def copy( x: Array[Double], y : Array[Double]) : Unit = f2jBLAS.dcopy(x.length, x, 1, y, 1)
}

object HasNetlibBlas extends Serializable {
  @transient private lazy val _f2jBLAS: NetlibBLAS = {
    initSparkBlas
    new F2jBLAS
  }

  private def initSparkBlas = synchronized {
    org.apache.spark.ml.linalg.BLAS.dot(Vectors.zeros(2), Vectors.zeros(2))
    org.apache.spark.ml.linalg.BLAS.gemv(1.0, Matrices.zeros(2, 2), Vectors.zeros(2), 0.5, Vectors.zeros(2).toDense)
  }

  @transient private lazy val _nativeBLAS: NetlibBLAS = {
    initSparkBlas
    NativeBLAS
  }
}
