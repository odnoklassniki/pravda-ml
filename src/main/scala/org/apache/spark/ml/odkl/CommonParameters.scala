package org.apache.spark.ml.odkl

/**
  * ml.odkl is an extension to Spark ML package with intention to
  * 1. Provide a modular structure with shared and tested common code
  * 2. Add ability to create train-only transformation (for better prediction performance)
  * 3. Unify extra information generation by the model fitters
  * 4. Support combined models with option for parallel training.
  *
  * This particular file contains common parameters sets used in multiple places.
  */

import org.apache.spark.ml.param._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructField
import org.json4s.DefaultWriters._
import org.json4s.jackson.JsonMethods
import org.json4s.{DefaultFormats, JValue, _}

/**
  * Adds parameter with column for instance type.
  */
trait HasTypeCol extends Params {

  final val typeColumn: Param[String] = new Param[String](
    this, "typeColumn", "Column where the value for type is stored (not null, single).")

  setDefault(typeColumn, "type")

  def setTypeColumn(value: String): this.type = set(typeColumn, value)
}

/**
  * Adds parameter with column for instance classes.
  */
trait HasClassesCol extends Params {
  final val classesColumn: Param[String] = new Param[String](
    this, "classesColumn", "Column where the value for classes is stored (nullable, single or seq).")

  setDefault(classesColumn, "classes")

  def setClassesColumn(value: String): this.type = set(classesColumn, value)
}

/**
  * Adds parameter wot classes weights (defaults to 1.0)
  */
trait HasClassesWeights extends Params {
  final val classesWeights = JacksonParam.mapParam[Double](
    this, "classesWeights", "Weights for the classes to combine (defaults to 1.0).")

  setDefault(classesWeights, Map[String, Double]())

  def setClassesWeights(weights: (String,Double)*) : this.type = set(classesWeights, weights.toMap)

  def getClassesWeights = $(classesWeights)
}

/**
  * Parameters for specifying which columns to include or exclude.
  */
trait HasColumnsSets extends Params {
  val columnsToInclude = new StringArrayParam(this, "columnsToInclude", "Columns to include into result. Mutually exclusive with columns to exclude.")
  val columnsToExclude = new StringArrayParam(this, "columnsToExclude", "Columns to exclude from result. Mutually exclusive with columns to include.")

  def setColumnsToInclude(columns: String*): this.type = set(columnsToInclude, columns.toArray)

  def setColumnsToExclude(columns: String*): this.type = set(columnsToExclude, columns.toArray)

  setDefault(columnsToExclude -> Array(), columnsToInclude -> Array())

  def extractColumns(dataset: DataFrame): Array[StructField] = {
    if ($(columnsToInclude).isEmpty && !$(columnsToExclude).isEmpty) {
      // Explicitly excluded
      val set = $(columnsToExclude).toSet
      dataset.schema.fields.filterNot(x => set.contains(x.name))
    } else if (!$(columnsToInclude).isEmpty && $(columnsToExclude).isEmpty) {
      // Explicitly included
      $(columnsToInclude).map(f => dataset.schema(f))
    } else if ($(columnsToInclude).isEmpty && $(columnsToExclude).isEmpty) {
      // Implicitly all
      dataset.schema.fields
    }
    else {
      // Can not do both include and exclude...
      throw new IllegalArgumentException("Only one of the columnsToInclude or columnsToExclude can be set")
    }
  }
}

/**
  * Adds parameter with the name of test/train split column
  */
trait HasIsTestCol extends Params {
  final val isTestColumn: Param[String] = new Param[String](
    this, "testMarker", "Boolean column with test instance marker.")

  setDefault(isTestColumn, "isTest")
}

/**
  * Adds parameters for folding - number of folds and name of column with fold number.
  */
trait HasFolds extends Params {
  final val numFolds: IntParam = new IntParam(
    this, "numFolds", "Number of folds to split data to.")

  final val numFoldsColumn: Param[String] = new Param[String](
    this, "numFoldsColumn", "Name of the column to store number of fold")

  setDefault(numFolds, 10)
  setDefault(numFoldsColumn, "foldNum")

  def setNumFolds(value: Int): this.type = set(numFolds, value)

  def setNumFoldsColumn(value: String): this.type = set(numFoldsColumn, value)
}

/**
  * For transformers performing grouping by a certain columns.
  */
trait HasGroupByColumns {
  this: Params =>

  final val groupByColumns = new StringArrayParam(
    this, "groupByColumns", "Grouping criteria for the evaluation.")

  def setGroupByColumns(columns: String*): this.type = set(groupByColumns, columns.toArray)
}

/**
  * For transformers performing sorting by a certain columns.
  */
trait HasSortByColumns {
  this: Params =>

  final val sortByColumns = new StringArrayParam(
    this, "sortByColumns", "Sorting criteria for the evaluation.")

  def setSortByColumns(columns: String*): this.type = set(sortByColumns, columns.toArray)
}

/**
  * Settings for partitioning, except the number of partitions. Is extended by static and dynamic partitioners.
  */
trait PartitioningParams extends Params with HasSortByColumns{

  val partitionBy = new StringArrayParam(this, "partitionBy", "Columns to partition dataset by")
  val sortBy = new StringArrayParam(this, "sortBy", "Columns to sort dataset by. Note that unlike Hadoop spark does " +
    "not sort partitions by default, thus you need to explicitly add partition by columns to sort by list")

  def setPartitionBy(columns:String*) : this.type = set(partitionBy, columns.toArray)
  def setSortBy(columns : String*) : this.type = set(sortBy, columns.toArray)
}

/**
  * For estimators capable of caching training data.
  */
trait HasCacheTrainData {
  this: Params =>

  final val cacheTrainData: BooleanParam = new BooleanParam(this, "cacheTrainData", "whether to cache dataset passed to optimizer")

  def setCacheTrainData(value: Boolean): this.type = set(cacheTrainData, value)

  /** @group getParam */
  final def getCacheTrainData: Boolean = $(cacheTrainData)
}

/**
  * Used for evaluators with batch support
  */
trait HasBatchSize extends Params {
  val batchSize = new  IntParam(this, "batchSize", "Amount of sample to put into batch before calculating the gradient.")

  def setBatchSize(value: Int): this.type = set(batchSize, value)
  def getBatchSize : Int = $(batchSize)
}

/**
  * Used to indicate that last weight should not be considered as a part of regularization (typically if it is
  * the intercept)
  */
trait HasRegularizeLast extends Params {

  val regularizeLast = new BooleanParam(this, "regularizeLast", "Whenever to regularize the last feature (should be set to false if last feature is an intercept).")

  def setRegularizeLast(value: Boolean): this.type = set(regularizeLast, value)
  def getRegularizeLast : Boolean = $(regularizeLast)
}

/**
  * For vector assemblers used to provide better naming for metadata attrbiutes.
  */
trait HasColumnAttributeMap {
  this: Params =>

  val columnAttributeMap = JacksonParam.mapParam[String](
    this, "columnAttributeMap", "Used to apply better naming for metadata attributes")

  def getColumnAttributeName(column: String): String = $(columnAttributeMap).getOrElse(column, column)

  def setColumnAttributeMap(value: (String, String)*): this.type = set(columnAttributeMap, value.toMap)

  setDefault(columnAttributeMap -> Map())
}
