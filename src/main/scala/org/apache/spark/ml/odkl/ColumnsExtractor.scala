package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains utilities for extrating multiple complex values from a dataframe.
  */

import org.apache.spark.SparkContext
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql._
import org.apache.spark.sql.odkl.SparkSqlUtils
import org.apache.spark.sql.types._
import org.json4s.DefaultWriters._
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, JValue}

/**
  * Used to extract a set of columns from the underlying data frame based on names and/or SQL expresions.
  */
class ColumnsExtractor(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("columnsExtractor"))

  val columnStatements = new Param[Seq[(String, String)]](
    this, "columnStatements", "Map with statements used to produce columns from the underlying data set.") {
    override def jsonEncode(value: Seq[(String, String)]): String = {
      val jValue: JValue = JsonMethods.render(JsonMethods.asJValue(value.toMap))
      JsonMethods.compact(jValue)
    }

    override def jsonDecode(json: String): Seq[(String, String)] = {
      implicit val formats = DefaultFormats
      JsonMethods.parse(json).extract[Map[String, String]].toSeq
    }
  }

  val columnMetadata = new Param[Map[String, Metadata]](
    this, "columnMetadata", "Map with statements used to produce columns from the underlying data set.") {
    override def jsonEncode(value: Map[String, Metadata]): String = {
      val jValue: JValue = JsonMethods.render(JsonMethods.asJValue(value.map(x => x._1 -> x._2.json)))
      JsonMethods.compact(jValue)
    }

    override def jsonDecode(json: String): Map[String, Metadata] = {
      implicit val formats = DefaultFormats
      JsonMethods.parse(json).extract[Map[String, String]].map(x => x._1 -> Metadata.fromJson(x._2))
    }
  }

  setDefault(columnStatements -> Seq(), columnMetadata -> Map())

  def withColumns(columns: String*): this.type =
    set(columnStatements, $(columnStatements) ++ columns.map(x => x -> x))

  def withExpresions(columns: (String, String)*): this.type =
    set(columnStatements, $(columnStatements) ++ columns)

  def withColumn(column: String, expression: String, metadata: Metadata): this.type = {
    set(columnStatements, $(columnStatements) :+ column -> expression)
    set(columnMetadata, $(columnMetadata) + (column -> metadata))
  }

  override def transform(dataset: DataFrame): DataFrame = select(dataset)

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    val sc = SparkContext.getOrCreate()
    val sqlContext = SQLContext.getOrCreate(sc)
    val dummyRDD = sc.parallelize(Seq[Row]())
    val dummyDF = SparkSqlUtils.reflectionLock.synchronized(sqlContext.createDataFrame(dummyRDD, schema))

    select(dummyDF).schema
  }

  def select(dataset: DataFrame) = {
    val columns: Seq[Column] = $(columnStatements).map {
      case (name, expr) =>
        require(name != null, "Null for column name not allowed")
        require(expr != null, s"Null expression for column $name not allowed")
        $(columnMetadata).get(name).map(
          // Try to add metadata from params
          m => functions.expr(expr).as(name, m))
          .getOrElse(
            // When check if there is a corresponding field in input
            dataset.schema.find(f => f.name.equals(expr)).map(f => functions.expr(expr).as(name, f.metadata))
              .getOrElse(
                // Finally, create field with no metadata
                functions.expr(expr).as(name)))
    }
    dataset.select(columns: _*)
  }
}

/**
  * Adds read ability.
  */
object ColumnsExtractor extends DefaultParamsReadable[ColumnsExtractor]