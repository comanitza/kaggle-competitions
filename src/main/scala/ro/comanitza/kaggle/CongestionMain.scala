package ro.comanitza.kaggle

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.{col, date_format, udf}
import org.joda.time.DateTime
import org.joda.time.format.DateTimeFormat

object CongestionMain extends SparkBase {

  /**
   *
   * Kaggle March 2022
   *
   * https://www.kaggle.com/c/tabular-playground-series-mar-2022
   *
   * @param args params
   */
  def main(args: Array[String]): Unit = {
    println("kaggle congestion competition - https://www.kaggle.com/c/tabular-playground-series-mar-2022")

    prepareEnv()

    testCongestion()
  }

  private def testCongestion(): Unit = {

    val session = createSparkSession("kaggle-competition-march-2022-congestion")

    val extractTimeToDouble = (dateAsString: String) => {

      val date = DateTime.parse(dateAsString, DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss"))

      date.hourOfDay().get().toDouble + (if (date.minuteOfHour().get() > 30) 0.5 else 0)
    }

    val udfExtractTimeToDouble = udf (extractTimeToDouble)

    val df = session.read.format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("D:\\stuff\\texts\\kaggle\\congestion\\train.csv")
      .withColumn("label", col("congestion"))
      .drop("congestion")
      .withColumn("week_day", date_format(col("time"), "E"))
      .withColumn("hourAsDouble", udfExtractTimeToDouble(col("time")))

    val testDf = session.read.format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("D:\\stuff\\texts\\kaggle\\congestion\\test.csv")
      .drop("congestion")
      .withColumn("week_day", date_format(col("time"), "E"))
      .withColumn("hourAsDouble", udfExtractTimeToDouble(col("time")))

    println("df count: " + df.count() + " and testDf count " + testDf.count())

    val directionIndexer = new StringIndexer()
      .setInputCol("direction")
      .setOutputCol("indexedDirection")

    val directionHotEncoder = new OneHotEncoder()
      .setInputCol("indexedDirection")
      .setOutputCol("hotDirection")

    val dayIndexer = new StringIndexer()
      .setInputCol("week_day")
      .setOutputCol("indexed_week_day")

    val dayOneHotEncoder = new OneHotEncoder()
      .setInputCol("indexed_week_day")
      .setOutputCol("hot_week_day")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("x", "y", "hourAsDouble", "hotDirection", "hot_week_day"))
      .setOutputCol("features")

    val regressor = new RandomForestRegressor()

    val evaluator = new RegressionEvaluator()
    val pipeline = new Pipeline().setStages(Array(directionIndexer, directionHotEncoder, dayIndexer, dayOneHotEncoder, vectorAssembler, regressor))

    val paramGrid = new ParamGridBuilder()
      .addGrid(regressor.maxBins, Array(16, 24))
      .addGrid(regressor.maxDepth, Array(16, 24, 30))
      .addGrid(regressor.numTrees, Array(20, 30))
      .build()

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)

    val Array(trainDataDf, testDataDf) = df.randomSplit(Array(0.9, 0.1))

    val model = crossValidator.fit(trainDataDf)

    val testPrediction = model.transform(testDataDf)

    testPrediction.show()

    val accuracy = evaluator.evaluate(testPrediction)

    println("accuracy: " + accuracy)

    model.transform(testDf).withColumn("congestion", col("prediction"))
      .select("row_id", "congestion")
      .coalesce(1)
      .write.format("csv")
      .option("header", "true")
      .save("D:\\stuff\\texts\\kaggle\\congestion\\results\\res-" + accuracy + "-" +System.currentTimeMillis())
  }
}
