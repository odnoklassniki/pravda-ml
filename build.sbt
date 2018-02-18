name := "odkl-ml-pipelines"

version := "1.0"

scalaVersion := "2.10.7"

// Have to fix Guava in order to avoid conflict on Stopwatch in FileInputFormat (of haddop 2.6.5)
libraryDependencies += "com.google.guava" % "guava" % "16.0.1" withSources()

libraryDependencies ++= {
  val sparkVer = "2.2.1"
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