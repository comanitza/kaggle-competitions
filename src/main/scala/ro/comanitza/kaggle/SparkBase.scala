package ro.comanitza.kaggle

import java.lang.management.ManagementFactory

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Encoder, SparkSession}

import scala.collection.mutable

/**
 *
 * Base abstract class to provided a placeholder for common methods
 *
 */
abstract class SparkBase {

  /**
   *
   * Method for creating a Apache Spark Session
   *
   * @param appName the name of the app
   * @return the created session
   */
  protected def createSparkSession(appName: String = "spark-ml-kaggle-competitions"): SparkSession = {

    val conf = new SparkConf().setAppName(appName).setMaster("local[4]")

    val session = SparkSession.builder().config(conf).getOrCreate()

    session.sparkContext.setLogLevel("WARN")
    println("create spark session version " + session.sparkContext.version)
    println("java version: " + ManagementFactory.getRuntimeMXBean.getSpecVersion)
    println("scala versions: " + util.Properties.versionString)

    session
  }

  /**
   * Method for setting the env prop to the needed file, is needed just on Windows systems, must change to appropiate file location
   */
  protected def prepareEnv(): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\stuff\\hadoopHome\\winutils-master\\winutils-master\\hadoop-3.0.0")
    System.setProperty("spark.local.dir", "D:\\temp\\hadoop")
    System.setProperty("java.io.tmpdir", "D:\\temp")
  }

  def allFeaturesCombinations(features: Array[String]): mutable.Set[Array[String]] = {

    val result = new mutable.HashSet[Array[String]]()

    for (i <- features.indices) {

      for (j <- i + 1 until features.length) {

        result.add(features.slice(i, j))
      }
    }

    result
  }

  protected def printRowsCountByValue(targetColumn: String, df: DataFrame)(implicit outputEncoder: Encoder[String]): Unit = {

    val targets = df.select(targetColumn).distinct().map(r => r.getString(0)).collect()

    println(s"### $targetColumn ### ### ###")
    for (t <- targets) {

      println(t + " -> " + df.where(s"$targetColumn='$t'").count() + " count")

    }
  }
}
