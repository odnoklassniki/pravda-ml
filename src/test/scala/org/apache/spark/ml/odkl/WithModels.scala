package org.apache.spark.ml.odkl

/**
  * Created by dmitriybugaichenko on 25.01.16.
  */
trait WithModels extends WithTestData {
  lazy val interceptedSgdModel = WithModels._interceptedSgdModel
  lazy val noInterceptLogisticModel = WithModels._noInterceptLogisticModel
  lazy val noInterceptLogisticRegularizedModel = WithModels._noInterceptLogisticRegularizedModel
  lazy val interceptLogisticModel = WithModels._interceptLogisticModel
}

object WithModels extends WithTestData {
  lazy val _interceptedSgdModel = {
    val model: LinearRegressionModel = Interceptor.intercept(new LinearRegressionSGD()).fit(interceptData)
    model
  }
  lazy val _noInterceptLogisticModel = {
    val model: LogisticRegressionModel = new LogisticRegressionLBFSG().fit(noInterceptDataLogistic)
    model
  }
  lazy val _noInterceptLogisticRegularizedModel = {
    val model: LogisticRegressionModel = new LogisticRegressionLBFSG().setRegParam(0.02).fit(noInterceptDataLogistic)
    model
  }
  lazy val _interceptLogisticModel = {
    val model = Interceptor.intercept(new LogisticRegressionLBFSG().setPreInitIntercept(true)).fit(interceptDataLogistig)
    model
  }
}
