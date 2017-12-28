package org.apache.spark.ml.odkl
import org.apache.spark.ml.param.{Param, Params}

/**
  * Created by dmitriybugaichenko on 30.11.16.
  */
trait HasDescriminantColumn extends Params {

  def descriminantColumn: Param[String]

  def getDescriminantColumn : String = this.$(descriminantColumn)
}
