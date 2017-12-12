package org.apache.spark.ml.odkl.texts

import java.util

import scala.collection.JavaConverters._

/**
  * Created by eugeny.malyutin on 28.07.16.
  */
object NGramUtils {
  def nGram(input: java.util.List[String], lowerNGramBound: Int, upperNGramBound: Int): util.List[String] = {
    nGramFun(input.asScala, lowerNGramBound, upperNGramBound).asJava
  }

  def nGramFun(input: Seq[String], lowerNGramBound: Int, upperNGramBound: Int): Seq[String] = {
    {
      for (i <- Range(lowerNGramBound, upperNGramBound + 1)) yield {
        //if upTo - Range(n,n+1) = List(n) e.g. standard  NGrams
        input.iterator.sliding(i).withPartial(false).map(_.mkString(" ")).toArray
      }
    }.flatMap(f => {
      f
    }).toSeq
  }
}
