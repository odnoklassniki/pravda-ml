package org.apache.spark.ml.odkl

import org.apache.spark.ml.linalg.{DenseMatrix, Matrix, VectorUDT}

/**
  * Created by dmitriybugaichenko on 19.11.16.
  *
  * Utility alowing access of certain hidden methods of Spark's mllib linalg
  */
object MatrixUtils {

  def vectorUDT = new VectorUDT()

  def transformDense(matrix: DenseMatrix, transformer: (Int, Int, Double) => Double): DenseMatrix = {
    matrix.foreachActive((i, j, v) => {
      matrix(i, j) = transformer(i, j, v)
    })
    matrix
  }

  def applyNonZeros(source: Matrix, target: DenseMatrix, transformer: (Int, Int, Double, Double) => Double): DenseMatrix = {
    source.foreachActive((i, j, v) => {
      val index = target.index(i, j)
      target.values(index) = transformer(i, j, v, target.values(index))
    })
    target
  }

  def applyAll(source: Matrix, target: DenseMatrix, transformer: (Int, Int, Double, Double) => Double): DenseMatrix = {
    for (j <- 0 until source.numCols; i <- 0 until source.numRows) {
      val index = target.index(i, j)
      target.values(index) = transformer(i, j, source(i, j), target.values(index))
    }
    target
  }
}
