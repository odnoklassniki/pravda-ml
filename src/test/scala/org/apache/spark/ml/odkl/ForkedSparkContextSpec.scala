package org.apache.spark.ml.odkl

import java.io.File

import breeze.linalg
import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.commons.io.FileUtils
import org.scalatest.FlatSpec

class ForkedSparkContextSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with WithTestData {

  "Fork " should " support one layer" in {
    val directory = new File(FileUtils.getTempDirectory, "forkedSpark")
    try {
      val estimator = new ForkedSparkEstimator[LinearRegressionModel,LinearRegressionSGD](new LinearRegressionSGD())
        .setTempPath(directory.getAbsolutePath)
        .setMaster("local[1]")

      val model = estimator.fit(noInterceptData)

      val dev: linalg.Vector[Double] = hiddenModel.asBreeze - model.getCoefficients.asBreeze

      val deviation: Double = dev dot dev

      deviation should be <= delta
      model.getIntercept should be(0.0)
    } finally {
      FileUtils.deleteDirectory(directory)
    }
  }

  "Fork " should " support two layers" in {
    val directory = new File(FileUtils.getTempDirectory, "forkedSpark")
    try {
      val estimator =  new ForkedSparkEstimator[LinearRegressionModel,ForkedSparkEstimator[LinearRegressionModel,LinearRegressionSGD]](
          new ForkedSparkEstimator[LinearRegressionModel,LinearRegressionSGD](new LinearRegressionSGD())
          .setTempPath(directory.getAbsolutePath)
          .setMaster("local[1]"))
        .setTempPath(directory.getAbsolutePath)
        .setMaster("local[1]")

      val model = estimator.fit(noInterceptData)

      val dev: linalg.Vector[Double] = hiddenModel.asBreeze - model.getCoefficients.asBreeze

      val deviation: Double = dev dot dev

      deviation should be <= delta
      model.getIntercept should be(0.0)
    } finally {
      FileUtils.deleteDirectory(directory)
    }
  }
}
