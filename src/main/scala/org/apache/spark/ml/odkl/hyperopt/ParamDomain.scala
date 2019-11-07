package org.apache.spark.ml.odkl.hyperopt

import org.apache.spark.ml.param.{Param, ParamMap, ParamPair}
import org.apache.spark.sql.Row

/**
  * Parameters domain is used to map from the [0,1] value sampled from the optimizer to the
  * actual parameter value.
  */
trait ParamDomain[T] {
  /**
    * Maps parameter value to [0,1]
    */
  def toDouble(domain: T) : Double

  /**
    * Maps sampled value from [0,1] to parameter value
    */
  def fromDouble(double: Double) : T

  /**
    * Used to indicate discrete parameters. Sampler discretize candidates
    */
  def numDiscreteValues : Option[Int]
}

/**
  * Holds the actual SparkML param and its domain. Support type-safe methods for moving data between optimizer,
  * data frame and SparkML estimator.
  */
case class ParamDomainPair[T](param: Param[T], domain: ParamDomain[T]) {
  def toDouble(paramMap: ParamMap) : Double = domain.toDouble(paramMap.get(param).get)

  def toParamPair(double: Double) : ParamPair[T] = ParamPair(param, domain.fromDouble(double))

  def toPairFromRow(row : Row, column: String) : ParamPair[T] = ParamPair(param, row.getAs[T](column))
}

/**
  * Models a simple real valued parameter from the range [lower,upper]
  */
case class DoubleRangeDomain(lower: Double, upper: Double) extends ParamDomain[Double] {
  override def toDouble(domain: Double): Double = (domain - lower) / (upper - lower)

  override def fromDouble(double: Double): Double = double * (upper - lower) + lower

  override def numDiscreteValues: Option[Int] = None
}

/**
  * Models a ordinal valued parameter from the sequence {lower, lower + 1, ... , upper}
  */
case class IntRangeDomain(lower: Int, upper: Int) extends ParamDomain[Int] {
  override def toDouble(domain: Int): Double = (domain.toDouble - lower.toDouble) / (upper.toDouble - lower.toDouble)

  override def fromDouble(double: Double): Int = (double * (upper - lower + 1)).toInt + lower

  override def numDiscreteValues: Option[Int] = Some(upper - lower + 1)
}

/**
  * Models parameter having limited set of values
  */
case class CategorialParam[T](values: Array[T]) extends ParamDomain[T] {
  override def toDouble(domain: T): Double = {
    val index = values.indexWhere(_.equals(domain))
    require(index >= 0, s"Failed to resolve domain value $domain")
    index.toDouble / values.length.toDouble
  }

  override def fromDouble(double: Double): T = values((double * values.length).toInt)

  override def numDiscreteValues: Option[Int] = Some(values.length)
}