package org.apache.spark.ml.odkl

import breeze.numerics.log2
import odkl.analysis.spark.TestEnv
import odkl.analysis.spark.util.SQLOperations
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.mllib.linalg.Vector
import org.scalatest.FlatSpec

/**
  * Created by dmitriybugaichenko on 25.01.16.
  */
class PartitionedRankingEvaluationsSpec extends FlatSpec with TestEnv with org.scalatest.Matchers with SQLOperations with WithModels with HasMetricsBlock {

  case class Prediction(userId: Long, objectType: String, ownerId: Long, score: Double, label: Double)

  lazy val data = sqlc.createDataFrame(Seq(
    Prediction(userId = 1, score = 0.9, label = 1.0, objectType = "Photo", ownerId = 1l),
    Prediction(userId = 1, score = 0.8, label = 0.0, objectType = "Post", ownerId = 2l),
    Prediction(userId = 1, score = 0.7, label = 1.0, objectType = "Post", ownerId = 3l),
    Prediction(userId = 1, score = 0.0, label = 1.0, objectType = "Post", ownerId = 3l)
    ,
    Prediction(userId = 2, score = 0.9, label = 1.0, objectType = "Photo", ownerId = 1l),
    Prediction(userId = 2, score = 0.8, label = 1.0, objectType = "Post", ownerId = 2l),
    Prediction(userId = 2, score = 0.3, label = 0.0, objectType = "Post", ownerId = 3l)
  ))

  lazy val evaluationsFrame = new PartitionedRankingEvaluator()
    .setGroupByColumns("userId")
    .setPredictionColumn("score")
    .setExtraColumns("objectType", "ownerId")
    .setMetrics(
      PartitionedRankingEvaluator.ndcgStrong(),
      PartitionedRankingEvaluator.ndcgStrongAt(2),
      PartitionedRankingEvaluator.ndcgWeak(),
      PartitionedRankingEvaluator.ndcgWeakAt(2),
      PartitionedRankingEvaluator.auc(),
      PartitionedRankingEvaluator.precision(),
      PartitionedRankingEvaluator.recall(),
      PartitionedRankingEvaluator.precisionAt(2),
      PartitionedRankingEvaluator.recallAt(2),
      PartitionedRankingEvaluator.numPositives(),
      PartitionedRankingEvaluator.numNegatives(),
      PartitionedRankingEvaluator.foundPositives(),
      PartitionedRankingEvaluator.foundNegatves(),
      PartitionedRankingEvaluator.f1(),
      PartitionedRankingEvaluator.f1At(2),
      PartitionedRankingEvaluator.countIf("photos", r => r.getString(2).equals("Photo")),
      PartitionedRankingEvaluator.countIf("posts", r => r.getString(2).equals("Post")),
      PartitionedRankingEvaluator.countIfAt("photosAt2", 2, r => r.getString(2).equals("Photo")),
      PartitionedRankingEvaluator.countIfAt("postsAt2", 2, r => r.getString(2).equals("Post")),
      PartitionedRankingEvaluator.countRelevantIf("relevantPosts", r => r.getString(2).equals("Post")),
      PartitionedRankingEvaluator.countRelevantIfAt("relevantPostAt2", 2, r => r.getString(2).equals("Post")),
      PartitionedRankingEvaluator.countDistinctIf("distinctOwners", _ => true, r => r.getLong(3)),
      PartitionedRankingEvaluator.countDistinctRelevantIf("distinctRelevantOwners", _ => true, r => r.getLong(3)),
      PartitionedRankingEvaluator.countDistinctIfAt("distinctOwnersAt", 2, _ => true, r => r.getLong(3)),
      PartitionedRankingEvaluator.countDistinctRelevantIfAt("distinctRelevantOwnersAt", 2, _ => true, r => r.getLong(3)),
      PartitionedRankingEvaluator.countDistinctIf("distinctPostOwners", r => r.getString(2).equals("Post"), r => r.getLong(3)),
      PartitionedRankingEvaluator.countDistinctRelevantIf("distinctRelevantPostOwners", r => r.getString(2).equals("Post"), r => r.getLong(3)),
      PartitionedRankingEvaluator.countDistinctIfAt("distinctPostOwnersAt", 2, r => r.getString(2).equals("Post"), r => r.getLong(3)),
      PartitionedRankingEvaluator.countDistinctRelevantIfAt("distinctRelevantPostOwnersAt", 2, r => r.getString(2).equals("Post"), r => r.getLong(3))
    )
    .transform(data)

  lazy val evaluations: Map[Long, Vector] =
    evaluationsFrame.map(r => r.getLong(0) -> r.getAs[Vector](1)).collect().toMap


  lazy val multiLabel = new AutoAssembler()
    .setColumnsToInclude("score", "label")
    .setOutputCol("labels")
    .fit(data)
    .transform(data)

  lazy val multiLabelFrame = new NameAssigner().setInputCols("label")
    .transform(
      new PartitionedRankingEvaluator()
        .setGroupByColumns("userId")
        .setPredictionColumn("score")
        .setLabelColumn("labels")
        .setExtraColumns("objectType", "ownerId")
        .setMetrics(PartitionedRankingEvaluator.ndcgStrong(), PartitionedRankingEvaluator.auc())
        .transform(multiLabel))

  lazy val multiLabelEvaluations: Map[(Long, String), Vector] =
    multiLabelFrame.map(r => (r.getLong(0), r.getString(2)) -> r.getAs[Vector](1)).collect().toMap


  lazy val multiScore = new AutoAssembler()
    .setColumnsToInclude("score", "label")
    .setOutputCol("scores")
    .fit(data)
    .transform(data)

  lazy val multiScoreFrame = new NameAssigner().setInputCols("score")
    .transform(new PartitionedRankingEvaluator()
      .setGroupByColumns("userId")
      .setPredictionColumn("scores")
      .setLabelColumn("label")
      .setExtraColumns("objectType", "ownerId")
      .setMetrics(PartitionedRankingEvaluator.ndcgStrong(), PartitionedRankingEvaluator.auc())
      .transform(multiScore))

  lazy val multiScoreEvaluations: Map[(Long, String), Vector] =
    multiScoreFrame.map(r => (r.getLong(0), r.getString(2)) -> r.getAs[Vector](1)).collect().toMap

  lazy val multiLabelScore = new AutoAssembler()
    .setColumnsToInclude("score", "label")
    .setOutputCol("scores")
    .fit(multiLabel)
    .transform(multiLabel)

  lazy val multiLabelScoreFrame = new NameAssigner().setInputCols("label", "score")
    .transform(new PartitionedRankingEvaluator()
      .setGroupByColumns("userId")
      .setPredictionColumn("scores")
      .setLabelColumn("labels")
      .setExtraColumns("objectType", "ownerId")
      .setMetrics(PartitionedRankingEvaluator.ndcgStrong(), PartitionedRankingEvaluator.auc())
      .transform(multiLabelScore))

  lazy val multiLabelScoreEvaluations: Map[(Long, String, String), Vector] =
    multiLabelScoreFrame.map(r => {
      (r.getLong(0), r.getString(2), r.getString(3)) -> r.getAs[Vector](1)
    }).collect().toMap


  "Evaluator " should " calculate ndgcStrong" in {

    val idcg = 1.0 + 1.0 / log2(3) + 1.0 / log2(4)

    evaluations(1)(0) should be((1.0 + 1.0 / log2(4)) / idcg)
    evaluations(2)(0) should be(1.0)
  }

  "Evaluator " should " calculate ndgcStrong at 2" in {

    val idcg = 1.0 + 1.0 / log2(3)

    evaluations(1)(1) should be(1.0 / idcg)
    evaluations(2)(1) should be(1.0)
  }

  "Evaluator " should " calculate ndgcWeak" in {

    val idcg = 2.0 + 1.0 / log2(3)

    evaluations(1)(2) should be((1.0 + 1.0 / log2(3)) / idcg)
    evaluations(2)(2) should be(1.0)
  }

  "Evaluator " should " calculate ndgcWeak at 2" in {

    val idcg = 2.0

    evaluations(1)(3) should be(1.0 / idcg)
    evaluations(2)(3) should be(1.0)
  }

  "Evaluator " should " calculate auc" in {

    evaluations(1)(4) should be(0.5)
    evaluations(2)(4) should be(1.0)
  }

  "Evaluator " should " calculate precision" in {

    evaluations(1)(5) should be(2.0 / 3.0)
    evaluations(2)(5) should be(1.0)
  }

  "Evaluator " should " calculate recall" in {

    evaluations(1)(6) should be(2.0 / 3.0)
    evaluations(2)(6) should be(1.0)
  }

  "Evaluator " should " calculate precision at 2" in {

    evaluations(1)(7) should be(0.5)
    evaluations(2)(7) should be(1.0)
  }

  "Evaluator " should " calculate recall at 2" in {

    evaluations(1)(8) should be(0.5)
    evaluations(2)(8) should be(1.0)
  }

  "Evaluator " should " calculate num positives" in {

    evaluations(1)(9) should be(3.0)
    evaluations(2)(9) should be(2.0)
  }

  "Evaluator " should " calculate num negatives" in {

    evaluations(1)(10) should be(1.0)
    evaluations(2)(10) should be(1.0)
  }

  "Evaluator " should " calculate found positives" in {

    evaluations(1)(11) should be(2.0)
    evaluations(2)(11) should be(2.0)
  }

  "Evaluator " should " calculate found negatives" in {

    evaluations(1)(12) should be(1.0)
    evaluations(2)(12) should be(0.0)
  }

  "Evaluator " should " calculate f1" in {

    evaluations(1)(13) should be((8.0 / 9.0) / (4.0 / 3.0))
    evaluations(2)(13) should be(1.0)
  }

  "Evaluator " should " calculate f1 at 2" in {

    evaluations(1)(14) should be(0.5)
    evaluations(2)(14) should be(1.0)
  }

  "Evaluator " should " count objects" in {

    evaluations(1)(15) should be(1.0)
    evaluations(1)(16) should be(2.0)
    evaluations(2)(15) should be(1.0)
    evaluations(2)(16) should be(1.0)
  }

  "Evaluator " should " count objects at 2" in {

    evaluations(1)(17) should be(1.0)
    evaluations(1)(18) should be(1.0)
    evaluations(2)(17) should be(1.0)
    evaluations(2)(18) should be(1.0)
  }

  "Evaluator " should " count relevant objects" in {

    evaluations(1)(19) should be(1.0)
    evaluations(1)(20) should be(0.0)
    evaluations(2)(19) should be(1.0)
    evaluations(2)(20) should be(1.0)
  }

  "Evaluator " should " count distinct owners" in {

    evaluations(1)(21) should be(3.0)
    evaluations(1)(22) should be(2.0)
    evaluations(2)(21) should be(2.0)
    evaluations(2)(22) should be(2.0)
  }

  "Evaluator " should " count distinct owners at 2" in {

    evaluations(1)(23) should be(2.0)
    evaluations(1)(24) should be(1.0)
    evaluations(2)(23) should be(2.0)
    evaluations(2)(24) should be(2.0)
  }

  "Evaluator " should " count distinct post owners" in {

    evaluations(1)(25) should be(2.0)
    evaluations(1)(26) should be(1.0)
    evaluations(2)(25) should be(1.0)
    evaluations(2)(26) should be(1.0)
  }

  "Evaluator " should " count distinct post owners at 2" in {

    evaluations(1)(27) should be(1.0)
    evaluations(1)(28) should be(0.0)
    evaluations(2)(27) should be(1.0)
    evaluations(2)(28) should be(1.0)
  }

  "Evaluator " should " calculate metrics for multiple labels" in {
    val idcg = 1.0 + 1.0 / log2(3) + 1.0 / log2(4)

    multiLabelEvaluations(1L -> "label")(0) should be((1.0 + 1.0 / log2(4)) / idcg)
    multiLabelEvaluations(1L -> "label")(1) should be(0.5)

    multiLabelEvaluations(1L -> "score")(0) should be(1.0)
    multiLabelEvaluations(1L -> "score")(1) should be(1.0)
  }

  "Evaluator " should " calculate metrics for multiple scores" in {
    val idcg = 1.0 + 1.0 / log2(3) + 1.0 / log2(4)

    multiScoreEvaluations(1L -> "score")(0) should be((1.0 + 1.0 / log2(4)) / idcg)
    multiScoreEvaluations(1L -> "score")(1) should be(0.5)

    multiScoreEvaluations(1L -> "label")(0) should be(1.0)
    multiScoreEvaluations(1L -> "label")(1) should be(1.0)
  }

  "Evaluator " should " calculate metrics for multiple labels and scores" in {
    val idcg = 1.0 + 1.0 / log2(3) + 1.0 / log2(4)

    multiLabelScoreEvaluations((1L, "score", "label"))(0) should be((1.0 + 1.0 / log2(4)) / idcg)
    multiLabelScoreEvaluations((1L, "score", "label"))(1) should be(0.5)

    multiLabelScoreEvaluations((1L, "label", "label"))(0) should be(1.0)
    multiLabelScoreEvaluations((1L, "label", "label"))(1) should be(1.0)

    multiLabelScoreEvaluations((1L, "score", "score"))(0) should be(1.0)
    multiLabelScoreEvaluations((1L, "score", "score"))(1) should be(1.0)

    val idcgInverse = (Math.pow(2, 0.9) - 1) + (Math.pow(2, 0.8) - 1) / log2(3) + (Math.pow(2, 0.7) - 1) / log2(4)

    multiLabelScoreEvaluations((1L, "label", "score"))(0) should be(((Math.pow(2, 0.9) - 1) + (Math.pow(2, 0.7) - 1) / log2(3)) / idcgInverse)
    multiLabelScoreEvaluations((1L, "label", "score"))(1) should be(2.0 / 3.0)
  }

  "Evaluator " should " add metadata" in {
    val attributes = AttributeGroup.fromStructField(evaluationsFrame.schema("metrics"))

    attributes.size should be(29)
    attributes.attributes.get.head.name.get should be("ndcg")
    attributes.attributes.get.last.name.get should be("distinctRelevantPostOwnersAt")
    attributes.attributes.get.map(_.name.get).toSet.size should be(29)
  }

  "Evaluator " should " add metadata for multilable" in {
    val attributes = AttributeGroup.fromStructField(multiLabelFrame.schema("metrics"))

    attributes.size should be(2)
    attributes.attributes.get.head.name.get should be("ndcg")
    attributes.attributes.get.last.name.get should be("auc")
    attributes.attributes.get.map(_.name.get).toSet.size should be(2)
  }
}
