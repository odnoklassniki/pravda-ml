package org.apache.spark.ml.odkl

import java.util.concurrent.ThreadLocalRandom
import java.util.regex.Pattern

import odkl.analysis.spark.util.Logging
import org.apache.spark.deploy.SparkSubmit
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.util.control.NonFatal

/**
  * This utility is used to support evaluation of the part of pipeline in a separate Spark app.
  * There are at least three identified use cases:
  * 1. Spark App with different settings for ETL and ML
  * 2. Support for larger fork factor in segmented hyperopt (scale driver if it became a bootleneck)
  * 3. Support for parallel XGBoost training (resolves internal conflict on the Rabbit part)
  *
  * Simple example with linear SGD and Zeppelin in yarn-client mode:
  *
  * {{{
  * // This estimator will start new Spark app from an app running in yarn-cluster mode
  * val secondLevel = new ForkedSparkEstimator[LinearRegressionModel, LinearRegressionSGD](new LinearRegressionSGD().setCacheTrainData(true))
  *             .setTempPath("tmp/forkedModels")
  *             // Match only files transfered with the app, re-point to the hdfs for faster start
  *             .withClassPathPropagation(".*__spark_libs__.*", ".+/" -> "hdfs://my-hadoop-nn/spark/lib/")
  *             // These files are localy available on all nodes
  *             .withClassPathPropagation("/opt/.*", "^/" -> "local://")
  *             // For convinience propagate configuration when working in non-interactive mode
  *             .setPropagateConfig(true)
  *             .setConfOverrides(
  *                 // Enable log aggregation and disable dynamic allocation
  *                 "spark.hadoop.yarn.log-aggregation-enable" -> "true",
  *                 "spark.dynamicAllocation.enabled" -> "false",
  *                 // These files might sneeak in when submited from Zeppelin, suppress them
  *                 "spark.yarn.dist.jars" -> "",
  *                 "spark.yarn.dist.files" -> "",
  *                 "spark.yarn.dist.archives" -> ""
  *                 )
  *             .setMaster("yarn")
  *             .setDeployMode("cluster")
  *             .setSubmitArgs(
  *                 "--num-executors", "1")
  *             .setName("secondLevel")
  *
  * // This estimator is will start neq Spark app from an interactive Zeppelin session
  * val firstLevel = new ForkedSparkEstimator[LinearRegressionModel, ForkedSparkEstimator[LinearRegressionModel,LinearRegressionSGD]](secondLevel)
  *         .setTempPath("tmp/forkedModels")
  *         // Propagate only odkl-analysiss jars, repoint to HDFS for faster start
  *         .withClassPathPropagation("/home/.*", ".+/" -> "hdfs://my-hadoop-nn/user/myuser/spark/lib/")
  *         // Do not propagate hell a lot of Zeppelin configs, rely on spark-defaults
  *         .setPropagateConfig(false)
  *         .setConfOverrides(
  *             // Enable log aggregation and disable dynamic execution
  *             "spark.hadoop.yarn.log-aggregation-enable" -> "true",
  *             "spark.dynamicAllocation.enabled" -> "false",
  *             // This is required to be able to start new spark apps from our app
  *             "spark.yarn.appMasterEnv.HADOOP_CONF_DIR" -> "/opt/hadoop/etc/hadoop/",
  *             // This is required to make sure Zeppelin does not full us the we are a Python app
  *             "spark.yarn.isPython" -> "false"
  *              )
  *         .setMaster("yarn")
  *         .setDeployMode("cluster")
  *         .setSubmitArgs(
  *             "--num-executors", "1")
  *         .setName("firstLevel")
  *
  *
  * val doubleForkedPipeiline = new Pipeline().setStages(Array(
  *     new VectorAssembler()
  *         .setInputCols(Array("first", "second"))
  *         .setOutputCol("features"),
  *     firstLevel
  *     ))
  * }}}
  */
class ForkedSparkEstimator[
M <: ModelWithSummary[M] with MLWritable,
E <: SummarizableEstimator[M] with MLWritable]
(
  override val uid: String
) extends SummarizableEstimator[M] with MLWritable {

  final val tempPath = new Param[String](this, "tempPath",
    "Where to store temporary data and model.")

  final val master = new Param[String](this, "master",
    "Type of the master for forked spark (same set as for spark_submit)")

  final val deployMode = new Param[String](this, "deployMode",
    "Type of the deployMode for forked spark (same set as for spark_submit)")

  final val name = new Param[String](this, "name",
    "Name of the application to submit.")

  final val confOverrides = JacksonParam.mapParam[String](
    this, "confOverrides",
    "Configuration for the forked spark process.")

  final val submitArgs = new StringArrayParam(this, "submitArgs",
    "Extra arguments to pass to spark submit.")

  final val mainJar = new Param[String](this, "mainJar",
    "Path to the main jar file to be used for the context. By default we are trying " +
      "to locate it from classpath.")

  final val classPathPropagations = JacksonParam.arrayParam[ClassPathExpression](this, "classPathPropagations",
    "Map with regexp filter for classpath records as a key and a sequence of reqexp path transformations (for " +
      "example, re-reference to the global storage in HDFS or local storage for faster start) " +
      "in pairs 'match' -> 'replacement'.")

  final val suppressConfig = new StringArrayParam(this, "suppressConfig",
    "List of configuration values to suppress from passing to the child context.")

  final val propagateConfig = new BooleanParam(this, "propagateConfig",
    "Whenever to propagate parent spark configuration to the submit. When used from the interactive environment" +
      " with valid SPARK_HOME and spark.defaults it might be better no to propagate config, while when working " +
      "on a cluster node it might be usefull.")

  setDefault(
    confOverrides -> Map(),
    submitArgs -> Array(),
    classPathPropagations -> Array(ClassPathExpression( filter = ".*", replacements = Array())),
    propagateConfig -> true,
    suppressConfig -> Array())

  def setTempPath(path: String): this.type = set(tempPath, path)

  def setConfOverrides(conf: (String, String)*): this.type = set(confOverrides, conf.toMap)

  def setSubmitArgs(args: String*): this.type = set(submitArgs, args.toArray)

  def withClassPathPropagation(filter: String, transformations: (String, String)*): this.type = {
    val newPart = ClassPathExpression(filter, transformations.toArray)
    if (isSet(classPathPropagations)) {
      set(classPathPropagations, $(classPathPropagations) :+ newPart)
    } else {
      set(classPathPropagations, Array(newPart))
    }

  }

  def setPropagateConfig(value: Boolean): this.type = set(propagateConfig, value)

  def setMaster(value: String): this.type = set(master, value)

  def setMainJar(value: String): this.type = set(mainJar, value)

  def setSuppressConfigs(name: String*): this.type = set(suppressConfig, name.toArray)

  def setDeployMode(value: String): this.type = set(deployMode, value)

  def setName(value: String): this.type = set(name, value)

  private var _nested: E = null.asInstanceOf[E]

  def nested: E = _nested

  def this(nested: E) = {
    this(Identifiable.randomUID("forkedSparkEstimator"))
    _nested = nested
  }

  private def setNested(nested: PipelineStage): this.type = {
    _nested = nested.asInstanceOf[E]
    this
  }

  override def copy(extra: ParamMap): ForkedSparkEstimator[M, E] =
    copyValues(new ForkedSparkEstimator[M, E](nested.copy(extra).asInstanceOf[E]))

  override def fit(dataset: Dataset[_]): M = {
    val random = Math.abs(ThreadLocalRandom.current().nextLong())

    val dataPath = $(tempPath) + s"/data_$uid-$random"
    val estimatorPath = $(tempPath) + s"/estimator_$uid-$random"
    val modelPath = $(tempPath) + s"/model_$uid-$random"

    val conf = dataset.sqlContext.sparkContext.conf

    val jars: Array[String] = if ($(classPathPropagations).nonEmpty) {
      val rawCP = System.getProperty("java.class.path").split(":")

      logInfo(s"Got class path files ${rawCP.mkString("[", ",", "]")}")

      $(classPathPropagations).flatMap(group => {
        val filter = group.filter
        val transformations = group.replacements

        val filteredCP = rawCP.filter(Pattern.compile(filter).asPredicate().test)

        val result = filteredCP.map(entry => transformations.foldLeft(entry)((e, pattern) => e.replaceAll(pattern._1, pattern._2))).distinct

        logInfo(s"For group $filter got classpath entries ${result.mkString("[", ",", "]")}")

        result
      }).toArray
    } else {
      Array[String]()
    }

    val pravdaMlPath = if (isDefined(mainJar)) {
      $(mainJar)
    } else {
      jars.find(_.contains("pravda-ml")).getOrElse(
        System.getProperty("java.class.path").split(":").find(_.contains("pravda-ml")).getOrElse(
          throw new IllegalArgumentException("PravdaML not found in classpath, specify the main jar directly")
        ))
    }

    logInfo(s"PravdaML jar resolved to $pravdaMlPath")

    val inheritedConfig = if ($(propagateConfig)) {
      conf.getAll.view
        .filterNot(x => $(confOverrides).contains(x._1))
        .filterNot(x => $(suppressConfig).contains(x._1))
        .flatMap(x => Seq("-c", s"${x._1}=${x._2}"))
        .toArray
    } else {
      Array[String]()
    }

    val customConfig = $(confOverrides).view
      .flatMap(x => Seq("-c", s"${x._1}=${x._2}"))


    val sparkJars = if (jars.nonEmpty) {
      val finallPath = jars.mkString(",")

      if ($(master).startsWith("yarn")) {
        Array("-c", s"spark.yarn.jars=$finallPath")
      } else {
        Array("--jars", finallPath)
      }
    } else {
      Array[String]()
    }

    val arguments = inheritedConfig ++ customConfig ++ $(submitArgs) ++ sparkJars ++
      get(deployMode).map(x => Array("--deploy-mode", x)).getOrElse(Array()) ++
      Array(
        "--master", $(master),
        "--name", get(name).getOrElse(uid),
        "--class", "org.apache.spark.ml.odkl.ForkedSparkEstimatorApp",
        pravdaMlPath,
        dataPath, estimatorPath, modelPath
      )

    logInfo(s"Got arguments: ${arguments.mkString("[", " ", "]")}")

    dataset.write.parquet(dataPath)
    nested.write.save(estimatorPath)

    @volatile var exception: Throwable = null

    val submissionThread = new Thread(new Runnable {
      override def run(): Unit = {

        try {

          SparkSubmit.main(arguments)

          logInfo("Spark submit finished.")
        } catch {
          case NonFatal(e) =>
            exception = e
            logError(s"Exception while fitting nested model: $e")
        }
      }
    })

    submissionThread.start()
    submissionThread.join()

    if (exception != null) {
      logError(s"Failed to build nested model: $exception")
      throw exception
    }

    DefaultParamsReader.loadParamsInstance(modelPath, dataset.sqlContext.sparkContext).asInstanceOf[M]
  }

  override def transformSchema(schema: StructType): StructType = nested.transformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override def save(path: String): Unit = {
      super.save(path)
      nested.write.save(s"$path/nested")
    }
  }
}

object ForkedSparkEstimator extends MLReadable[ForkedSparkEstimator[_, _]] {
  override def read: MLReader[ForkedSparkEstimator[_, _]] = new DefaultParamsReader[ForkedSparkEstimator[_, _]] {
    override def load(path: String): ForkedSparkEstimator[_, _] = {
      val as = super.load(path)
      val nested: PipelineStage = DefaultParamsReader.loadParamsInstance(s"$path/nested", sc)

      as.setNested(nested)
    }
  }
}

case class ClassPathExpression(filter: String, replacements: Array[(String,String)])

object ForkedSparkEstimatorApp extends App with Logging {
  try {
    // Note that for local mode this will be the same session.
    val sc = SparkSession.builder().getOrCreate()
    val sqlc = sc.sqlContext

    val dataPath = args(0)
    val estimatorPath = args(1)
    val modelPath = args(2)

    logInfo(s"Loading estimator from $estimatorPath")
    val pipeline = DefaultParamsReader.loadParamsInstance(estimatorPath, sc.sparkContext).asInstanceOf[SummarizableEstimator[_]]

    logInfo(s"Loading data from $dataPath")
    val data = sqlc.read.parquet(dataPath)

    logInfo("Fitting model")
    val model = pipeline.fit(data)

    logInfo(s"Saving model to $modelPath")
    model.asInstanceOf[MLWritable].write.save(modelPath)
  } catch {
    case e: Throwable => logError(s"Something went wrong: $e")
      throw e
  }
}