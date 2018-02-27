import sbt.Developer

name := "ok-ml-pipelines"

libraryDependencies ++= {
  val sparkVer = "1.6.3"
  Seq(
    "org.apache.spark"     %% "spark-core"              % sparkVer withSources(),
    "org.apache.spark"     %% "spark-mllib"             % sparkVer withSources(),
    "org.apache.spark"     %% "spark-sql"               % sparkVer withSources(),
    "org.apache.spark"     %% "spark-streaming"         % sparkVer withSources(),

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

    "com.optimaize.languagedetector" % "language-detector"  % "0.6",
    "com.tdunning" % "t-digest" % "3.2"
  )
}

organization := "ru.odnoklassniki"

version := "0.1-spark1.6-SNAPSHOT"

scalaVersion := "2.10.7"

crossScalaVersions := Seq("2.11.11")

licenses := Seq("Apache 2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0"))

homepage := Some(url("https://github.com/odnoklassniki/ok-ml-pipelines"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/odnoklassniki/ok-ml-pipelines"),
    "scm:git@github.com:odnoklassniki/ok-ml-pipelines.git"
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
