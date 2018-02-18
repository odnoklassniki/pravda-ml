package org.apache.spark.ml.odkl


import java.io.File

import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, BinaryAttribute, NumericAttribute}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Row, functions}
import org.scalatest.FlatSpec


/**
  * Created by vyacheslav.baranov on 02/12/15.
  */
class FeatureEncodersSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations
  with WithTestData with Serializable {

  sqlc.udf.register("vectorize", (r: Row) => Vectors.dense(r.getDouble(0), r.getDouble(1)))

  private val extender: ColumnsExtractor = new ColumnsExtractor()
    .withColumns("label", "first", "second")
    .withColumn(
      "vector",
      "vectorize(struct(first,second))",
      new AttributeGroup("vector", Array[Attribute](
        NumericAttribute.defaultAttr.withName("v1"),
        NumericAttribute.defaultAttr.withName("v2")
      )).toMetadata())
    .withExpresions("boolean" -> "CASE WHEN first > 0 THEN true ELSE false END")

  lazy val extendedData = extender.transform(rawData)

  private val assembler: AutoAssembler = new AutoAssembler()
    .setColumnsToExclude("label")
    .setOutputCol("features")

  lazy val transformed = assembler
    .fit(extendedData)
    .transform(extendedData)

  val nominalize = functions.udf[String, Row](
    r => if (r.getDouble(0) > r.getDouble(1)) "first" else "second")

  lazy val withNominal = extendedData.withColumn("nominal", nominalize(functions.struct("first", "second")))

  private val multinominalExtractor: MultinominalExtractor = new MultinominalExtractor()
    .setInputCol("nominal")
    .setOutputCol("nominal")

  lazy val withNominalTransformed = multinominalExtractor
    .fit(withNominal)
    .transform(withNominal)


  sqlc.udf.register("nominalize", (first: Double, second: Double) => (if (first > 0) Seq("first") else Seq()) ++ (if (second > 0) Seq("second") else Seq()))

  sqlc.udf.register("nominalize_s", (first: Double, second: Double) => if (first > second - 0.1) "first_s" else {if (second > first - 0.1) "second_s" else null})

  val withMultiNominalExtender = extender
    .copy(ParamMap())
    .withExpresions("nominal_s" -> "nominalize_s(first,second)")
    .withExpresions("nominal" -> "nominalize(first,second)")

  lazy val withMultiNominal = withMultiNominalExtender.transform(rawData)

  lazy val withMultiNominalTransformed = multinominalExtractor
    .fit(withMultiNominal)
    .transform(withMultiNominal)


  lazy val pipelineModel = new Pipeline().setStages(Array(
    withMultiNominalExtender,
    assembler.asInstanceOf[PipelineStage]
  )).fit(rawData)

  lazy val pipelineTransfom = pipelineModel.transform(rawData)

  lazy val reReadModel = {
    val directory = new File(FileUtils.getTempDirectory, pipelineModel.uid)
    try {
      pipelineModel.save(directory.getAbsolutePath)
      PipelineModel.read.context(sqlc).load(directory.getAbsolutePath)
    } finally {
      FileUtils.deleteDirectory(directory)
    }
  }

  lazy val reReadTransform = reReadModel.transform(rawData)

  "AutoAssembler " should " add scalar columns" in {
    transformed.select("first", "second", "features").rdd.map(r => (r.getDouble(0), r.getDouble(1), r.getAs[Vector](2))).collect.foreach {
      case (first: Double, second: Double, vector: Vector) =>
        vector(0) should be(first)
        vector(1) should be(second)
    }
  }

  "AutoAssembler " should " add binary columns" in {
    transformed.select("boolean", "features").rdd.map(r => (r.getBoolean(0), r.getAs[Vector](1))).collect.foreach {
      case (boolean: Boolean, vector: Vector) =>
        vector(4) should be(if (boolean) 1.0 else 0.0)
    }
  }

  "AutoAssembler " should " add NaN for booleans" in {
    val data = extendedData.withColumn(
      "boolean",
      functions.expr("CASE WHEN first > 0.5 THEN null ELSE boolean END"))

    assembler.fit(data).transform(data).select("features").rdd.map(r => r.getAs[Vector](0)).collect.foreach {
      vector =>
        if (vector(0) > 0.5) {
          vector(4).isNaN should be(true)
        } else if (vector(0) > 0) {
          vector(4) should be(1.0)
        } else {
          vector(4) should be(0.0)
        }
    }
  }

  "AutoAssembler " should " add NaN for scalars" in {
    val data = extendedData.withColumn(
      "second",
      functions.expr("CASE WHEN first > 0.5 THEN null ELSE second END"))

    assembler.fit(data).transform(data).select("features").rdd.map(r => r.getAs[Vector](0)).collect.foreach {
      vector =>
        if (vector(0) > 0.5) {
          vector(1).isNaN should be(true)
        } else {
          vector(1) should be(vector(3))
        }
    }
  }

  "AutoAssembler " should " add NaN for vectors" in {
    val data = extendedData.withColumn(
      "vector",
      functions.expr("CASE WHEN first > 0.5 THEN null ELSE vector END"))

    assembler.fit(data).transform(data).select("features").rdd.map(r => r.getAs[Vector](0)).collect.foreach {
      vector =>
        if (vector(0) > 0.5) {
          vector(2).isNaN should be(true)
          vector(3).isNaN should be(true)
        } else {
          vector(2) should be(vector(0))
          vector(3) should be(vector(1))
        }
    }
  }

  "AutoAssembler " should " add metadata" in {
    val attributes = AttributeGroup.fromStructField(transformed.schema.fields.last)

    attributes(0) should be(NumericAttribute.defaultAttr.withName("first").withIndex(0))
    attributes(1) should be(NumericAttribute.defaultAttr.withName("second").withIndex(1))
    attributes(2) should be(NumericAttribute.defaultAttr.withName("vector_v1").withIndex(2))
    attributes(3) should be(NumericAttribute.defaultAttr.withName("vector_v2").withIndex(3))
    attributes(4) should be(NumericAttribute.defaultAttr.withName("boolean").withIndex(4))
  }

  "MultiNominalExtractor " should " set values for single" in {
    withNominalTransformed
      .select("first", "second", "nominal").rdd
      .map(r => (r.getDouble(0), r.getDouble(1), r.getAs[Vector](2))).collect.foreach {
      case (first: Double, second: Double, vector: Vector) =>
        vector(1) should be(if (first > second) 1.0 else 0.0)
        vector(0) should be(if (second >= first) 1.0 else 0.0)
    }
  }

  "MultiNominalExtractor " should " add metadata for single" in {
    val attributes = AttributeGroup.fromStructField(withNominalTransformed.schema.fields.last)

    attributes(0) should be(BinaryAttribute.defaultAttr.withName("second").withIndex(0))
    attributes(1) should be(BinaryAttribute.defaultAttr.withName("first").withIndex(1))
  }

  "MultiNominalExtractor " should " set values for multiple" in {
    withMultiNominalTransformed
      .select("first", "second", "nominal").rdd
      .map(r => (r.getDouble(0), r.getDouble(1), r.getAs[Vector](2))).collect.foreach {
      case (first: Double, second: Double, vector: Vector) =>
        vector(0) should be(if (first > 0) 1.0 else 0.0)
        vector(1) should be(if (second > 0) 1.0 else 0.0)
    }
  }

  "MultiNominalExtractor " should " tolerate nulls" in {

    val nullify = functions.udf[Seq[String], Seq[String]](x => if (x.isEmpty) null else x)
    val withNulls = withMultiNominal.withColumn(
      "nominal", nullify(withMultiNominal("nominal")))

    multinominalExtractor.fit(withNulls).transform(withNulls)
      .select("first", "second", "nominal").rdd
      .map(r => (r.getDouble(0), r.getDouble(1), r.getAs[Vector](2))).collect.foreach {
      case (first: Double, second: Double, vector: Vector) =>
        vector(0) should be(if (first > 0) 1.0 else 0.0)
        vector(1) should be(if (second > 0) 1.0 else 0.0)
    }
  }

  "MultiNominalExtractor " should " add metadata  for multiple" in {
    val attributes = AttributeGroup.fromStructField(withMultiNominalTransformed.schema.fields.last)

    attributes(0) should be(BinaryAttribute.defaultAttr.withName("first").withIndex(0))
    attributes(1) should be(BinaryAttribute.defaultAttr.withName("second").withIndex(1))
  }

  "Pipelined model " should " set values" in {
    val attributes = AttributeGroup.fromStructField(pipelineTransfom.schema.fields.last)

    pipelineTransfom.select("first", "second", "vector", "boolean", "nominal_s", "nominal", "features").rdd.collect().foreach {
      case Row(first: Double, second: Double, vector: Vector, boolean: Boolean, nominal_s: Vector, nominal: Vector, features: Vector) =>
        features(0) should be(first)
        features(1) should be(second)
        features(2) should be(vector(0))
        features(3) should be(vector(1))
        features(4) should be(if (boolean) 1.0 else 0.0)
        features(5) should be(nominal_s(0))
        features(6) should be(nominal_s(1))
        features(7) should be(nominal(0))
        features(8) should be(nominal(1))
    }
  }

  "Pipelined model " should " add metadata" in {
    val attributes = AttributeGroup.fromStructField(pipelineTransfom.schema.fields.last)

    attributes(0) should be(NumericAttribute.defaultAttr.withName("first").withIndex(0))
    attributes(1) should be(NumericAttribute.defaultAttr.withName("second").withIndex(1))
    attributes(2) should be(NumericAttribute.defaultAttr.withName("vector_v1").withIndex(2))
    attributes(3) should be(NumericAttribute.defaultAttr.withName("vector_v2").withIndex(3))
    attributes(4) should be(NumericAttribute.defaultAttr.withName("boolean").withIndex(4))
    attributes(5) should be(BinaryAttribute.defaultAttr.withName("nominal_s_first_s").withIndex(5))
    attributes(6) should be(BinaryAttribute.defaultAttr.withName("nominal_s_second_s").withIndex(6))
    attributes(7) should be(BinaryAttribute.defaultAttr.withName("nominal_first").withIndex(7))
    attributes(8) should be(BinaryAttribute.defaultAttr.withName("nominal_second").withIndex(8))
  }

  "Pipelined model " should " set values after re-read" in {
    val attributes = AttributeGroup.fromStructField(reReadTransform.schema.fields.last)

    reReadTransform.select("first", "second", "vector", "boolean", "nominal_s", "nominal", "features").rdd.collect().foreach {
      case Row(first: Double, second: Double, vector: Vector, boolean: Boolean, nominal_s: Vector, nominal: Vector, features: Vector) =>
        features(0) should be(first)
        features(1) should be(second)
        features(2) should be(vector(0))
        features(3) should be(vector(1))
        features(4) should be(if (boolean) 1.0 else 0.0)
        features(5) should be(nominal_s(0))
        features(6) should be(nominal_s(1))
        features(7) should be(nominal(0))
        features(8) should be(nominal(1))
    }
  }

  "Pipelined model " should " add metadata after re-read" in {
    val attributes = AttributeGroup.fromStructField(reReadTransform.schema.fields.last)

    attributes(0) should be(NumericAttribute.defaultAttr.withName("first").withIndex(0))
    attributes(1) should be(NumericAttribute.defaultAttr.withName("second").withIndex(1))
    attributes(2) should be(NumericAttribute.defaultAttr.withName("vector_v1").withIndex(2))
    attributes(3) should be(NumericAttribute.defaultAttr.withName("vector_v2").withIndex(3))
    attributes(4) should be(NumericAttribute.defaultAttr.withName("boolean").withIndex(4))
    attributes(5) should be(BinaryAttribute.defaultAttr.withName("nominal_s_first_s").withIndex(5))
    attributes(6) should be(BinaryAttribute.defaultAttr.withName("nominal_s_second_s").withIndex(6))
    attributes(7) should be(BinaryAttribute.defaultAttr.withName("nominal_first").withIndex(7))
    attributes(8) should be(BinaryAttribute.defaultAttr.withName("nominal_second").withIndex(8))
  }

  "NaNToMeanReplacer " should " replace NaN to mean" in {
    val data = extendedData
      .withColumn("boolean", functions.expr("CASE WHEN first > 0.5 THEN null ELSE boolean END"))


    val assembled: DataFrame = assembler
      .fit(data)
      .transform(data)
      .withColumn("group", functions.expr("CASE WHEN second > 0.0 THEN 'Post' ELSE 'Photo' END"))

    val replaced = new NaNToMeanReplacerEstimator()
      .setGroupByColumn("group")
      .setInputCol("features")
      .setOutputCol("features")
      .fit(assembled)
      .transform(assembled)

    replaced.select("features").rdd.map(r => r.getAs[Vector](0)).collect.foreach {
      vector =>
        if (vector(0) > 0.5) {
          vector(4) should be((1.0 / 3) +- 0.03)
        } else if (vector(0) > 0) {
          vector(4) should be(1.0)
        } else {
          vector(4) should be(0.0)
        }
    }
  }

  "NaNToMeanReplacer " should " replace NaN to mean after re-read" in {
    val data = extendedData
      .withColumn("boolean", functions.expr("CASE WHEN first > 0.5 THEN null ELSE boolean END"))


    val assembled: DataFrame = assembler
      .fit(data)
      .transform(data)
      .withColumn("group", functions.expr("CASE WHEN second > 0.0 THEN 'Post' ELSE 'Photo' END"))

    val model = new Pipeline()
      .setStages(Array(new NaNToMeanReplacerEstimator()
        .setGroupByColumn("group")
        .setInputCol("features")
        .setOutputCol("features")))
      .fit(assembled)

    val reReadModel = {
      val directory = new File(FileUtils.getTempDirectory, model.uid)
      try {
        model.save(directory.getAbsolutePath)
        PipelineModel.read.context(sqlc).load(directory.getAbsolutePath)
      } finally {
        FileUtils.deleteDirectory(directory)
      }
    }

    val replaced = reReadModel
      .transform(assembled)

    replaced.select("features").rdd.map(r => r.getAs[Vector](0)).collect.foreach {
      vector =>
        if (vector(0) > 0.5) {
          vector(4) should be((1.0 / 3) +- 0.03)
        } else if (vector(0) > 0) {
          vector(4) should be(1.0)
        } else {
          vector(4) should be(0.0)
        }
    }
  }

}
