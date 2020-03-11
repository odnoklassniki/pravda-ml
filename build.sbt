import sbt.Developer

name := "pravda-ml"

// Have to fix Guava in order to avoid conflict on Stopwatch in FileInputFormat (of haddop 2.6.5)
libraryDependencies += "com.google.guava" % "guava" % "16.0.1" withSources()

libraryDependencies ++= {
  val sparkVer = "2.3.4"
  Seq(
    "org.apache.spark"     %% "spark-core"              % sparkVer withSources() exclude("com.google.guava", "guava"),
    "org.apache.spark"     %% "spark-mllib"             % sparkVer withSources() exclude("com.google.guava", "guava"),
    "org.apache.spark"     %% "spark-sql"               % sparkVer withSources() exclude("com.google.guava", "guava"),
    "org.apache.spark"     %% "spark-streaming"         % sparkVer withSources() exclude("com.google.guava", "guava"),

    "com.esotericsoftware" % "kryo" % "4.0.1"
  )
}

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.4" % Test,
  "org.mockito" % "mockito-core" % "2.13.0" % Test
)

libraryDependencies ++= {
  val luceneVersion = "5.4.1"
  Seq(
    "org.apache.lucene"    % "lucene-core"             % luceneVersion,
    "org.apache.lucene"    % "lucene-analyzers-common" % luceneVersion,

    "com.optimaize.languagedetector" % "language-detector"  % "0.6" exclude("com.google.guava", "guava"),
    "com.tdunning" % "t-digest" % "3.2"
  )
}

libraryDependencies ++= Seq(
  "ml.dmlc" % "xgboost4j" % "0.81" withSources(),
  "ml.dmlc" % "xgboost4j-spark" % "0.81" withSources()
)

libraryDependencies += "org.mlflow" % "mlflow-client" % "1.2.0"

organization := "ru.odnoklassniki"

version := "0.6.1-spark2.3"

scalaVersion := "2.11.8"

crossScalaVersions := Seq("2.11.11")

licenses := Seq("Apache 2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0"))

homepage := Some(url("https://github.com/odnoklassniki/pravda-ml"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/odnoklassniki/pravda-ml"),
    "scm:git@github.com:odnoklassniki/pravda-ml.git"
  )
)

developers := List(
  Developer(
    id    = "DmitryBugaychenko",
    name  = "Dmitry Bugaychenko",
    email = "dmitry.bugaychenko@corp.mail.ru",
    url   = url("https://www.linkedin.com/comm/in/dmitrybugaychenko")
  ),
  Developer(
    id    = "EugenyMalyutin",
    name  = "Eugeny Malyutin",
    email = "eugeny.malyutin@corp.mail.ru",
    url   = url("https://github.com/WarlockTheGrait")
  ),
  Developer(
    id    = "EugenyZhurin",
    name  = "Eugeny Zhurin",
    email = "eugeny.zhurin@corp.mail.ru",
    url   = url("https://github.com/Nordsvich")
  )
)

publishMavenStyle := true

pomIncludeRepository := { _ => false }

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

credentials += Credentials(Path.userHome / ".sonatype" / "credentials.ini")

useGpg := true
