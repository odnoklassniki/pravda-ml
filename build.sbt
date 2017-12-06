name := "odkl-ml-pipelines"

version := "1.0"

scalaVersion := "2.11.11"

libraryDependencies ++= {
  val sparkVer = "1.6.3"
  Seq(
    "org.apache.spark"     %% "spark-core"              % sparkVer withSources(),
    "org.apache.spark"     %% "spark-mllib"             % sparkVer withSources(),
    "org.apache.spark"     %% "spark-sql"               % sparkVer withSources(),
    "org.apache.spark"     %% "spark-streaming"         % sparkVer withSources()
  )
}