package ro.comanitza.kaggle

import java.lang.management.ManagementFactory

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

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
  }
}
