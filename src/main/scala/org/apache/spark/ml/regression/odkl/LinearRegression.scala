package org.apache.spark.ml.regression.odkl

import org.apache.spark.annotation.Since
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.odkl._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression
import org.apache.spark.ml.regression.LinearRegressionParams
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.util.Try

/**
  * Simple wrapper around the SparkML linear regression used to attach summary blocks.
  * TODO: Add unit tests
  *
  * @param uid
  */
class LinearRegression(override val uid: String) extends SummarizableEstimator[LinearRegressionModel]
with LinearRegressionParams with DefaultParamsWritable with HasWeights{

  def this() = this(Identifiable.randomUID("linearRegressor"))

  /**
    * Set the regularization parameter.
    * Default is 0.0.
    *
    * @group setParam
    */
  @Since("1.3.0")
  def setRegParam(value: Double): this.type = set(regParam, value)

  /**
    * Set if we should fit the intercept.
    * Default is true.
    *
    * @group setParam
    */
  @Since("1.5.0")
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)

  /**
    * Whether to standardize the training features before fitting the model.
    * The coefficients of models will be always returned on the original scale,
    * so it will be transparent for users.
    * Default is true.
    *
    * @note With/without standardization, the models should be always converged
    * to the same solution when no regularization is applied. In R's GLMNET package,
    * the default behavior is true as well.
    *
    * @group setParam
    */
  @Since("1.5.0")
  def setStandardization(value: Boolean): this.type = set(standardization, value)

  /**
    * Set the ElasticNet mixing parameter.
    * For alpha = 0, the penalty is an L2 penalty.
    * For alpha = 1, it is an L1 penalty.
    * For alpha in (0,1), the penalty is a combination of L1 and L2.
    * Default is 0.0 which is an L2 penalty.
    *
    * Note: Fitting with huber loss only supports None and L2 regularization,
    * so throws exception if this param is non-zero value.
    *
    * @group setParam
    */
  @Since("1.4.0")
  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)

  /**
    * Set the maximum number of iterations.
    * Default is 100.
    *
    * @group setParam
    */
  @Since("1.3.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
    * Set the convergence tolerance of iterations.
    * Smaller value will lead to higher accuracy with the cost of more iterations.
    * Default is 1E-6.
    *
    * @group setParam
    */
  @Since("1.4.0")
  def setTol(value: Double): this.type = set(tol, value)

  /**
    * Whether to over-/under-sample training instances according to the given weights in weightCol.
    * If not set or empty, all instances are treated equally (weight 1.0).
    * Default is not set, so all instances have weight one.
    *
    * @group setParam
    */
  @Since("1.6.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /**
    * Set the solver algorithm used for optimization.
    * In case of linear regression, this can be "l-bfgs", "normal" and "auto".
    *  - "l-bfgs" denotes Limited-memory BFGS which is a limited-memory quasi-Newton
    *    optimization method.
    *  - "normal" denotes using Normal Equation as an analytical solution to the linear regression
    *    problem.  This solver is limited to `LinearRegression.MAX_FEATURES_FOR_NORMAL_SOLVER`.
    *  - "auto" (default) means that the solver algorithm is selected automatically.
    *    The Normal Equations solver will be used when possible, but this will automatically fall
    *    back to iterative optimization methods when needed.
    *
    * Note: Fitting with huber loss doesn't support normal solver,
    * so throws exception if this param was set with "normal".
    * @group setParam
    */
  @Since("1.6.0")
  def setSolver(value: String): this.type = set(solver, value)

  /**
    * Suggested depth for treeAggregate (greater than or equal to 2).
    * If the dimensions of features or the number of partitions are large,
    * this param could be adjusted to a larger size.
    * Default is 2.
    *
    * @group expertSetParam
    */
  @Since("2.1.0")
  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)

  /**
    * Sets the value of param [[loss]].
    * Default is "squaredError".
    *
    * @group setParam
    */
  @Since("2.3.0")
  def setLoss(value: String): this.type = set(loss, value)

  /**
    * Sets the value of param [[epsilon]].
    * Default is 1.35.
    *
    * @group setExpertParam
    */
  @Since("2.3.0")
  def setEpsilon(value: Double): this.type = set(epsilon, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /** @group setParam */
  def setFeaturesCol(value: String): this.type  = set(featuresCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type  = set(predictionCol, value)

  override def copy(extra: ParamMap): SummarizableEstimator[LinearRegressionModel] = defaultCopy(extra)

  def extractWeights(
                      model: regression.LinearRegressionModel,
                      sparkSession: SparkSession,
                      attributes: AttributeGroup): DataFrame = {
    import sparkSession.sqlContext.implicits._

    val names = attributes.attributes.map(x => {
      x.map(a => a.index.get -> a.name.orNull).toMap
    }).getOrElse(Map())

    val pValues: Option[Array[Double]] = if (model.hasSummary) Try(model.summary.pValues).toOption else None

    val mayBeIntercept: Array[(Int, String, Double, Double)] =
      if (model.getFitIntercept) Array((model.coefficients.size, "#intercept", model.intercept, pValues.map(_.last).getOrElse(Double.NaN))) else Array()

    val summary =
      model.coefficients.toArray.zipWithIndex.map(x => (
        x._2,
        names.get(x._2).orNull,
        x._1,
        pValues.map(_(x._2)).getOrElse(Double.NaN))) ++
        mayBeIntercept

    SparkSqlUtils.reflectionLock.synchronized(
      sparkSession.sparkContext.parallelize(summary.toSeq, 1).toDF(index, name, weight, "pValue")
    )
  }

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    val model: regression.LinearRegressionModel = copyValues(new org.apache.spark.ml.regression.LinearRegression()).fit(dataset)

    val result: LinearModel[LinearRegressionModel] = copyValues(LinearRegressionModel.create(
      model.coefficients, dataset.sparkSession.sqlContext, dataset.schema($(featuresCol))))

    result.copy(
      params = ParamMap(
        result.coefficients -> model.coefficients,
        result.intercept -> model.intercept
      ),
      blocks = Map(
         weights -> extractWeights(model, dataset.sparkSession, AttributeGroup.fromStructField(dataset.schema($(featuresCol))))
      )
    )
  }

  override def transformSchema(schema: StructType): StructType =
    copyValues(new org.apache.spark.ml.regression.LinearRegression()).transformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]