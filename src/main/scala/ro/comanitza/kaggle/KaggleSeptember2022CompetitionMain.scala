package ro.comanitza.kaggle

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Encoder, SparkSession}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, desc, udf}
import org.joda.time.DateTime

//https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/overview
//https://towardsdatascience.com/effortless-hyperparameters-tuning-with-apache-spark-20ff93019ef2
//https://www.kaggle.com/code/param302/tps-eda-model/
object KaggleSeptember2022CompetitionMain extends SparkBase {

  private val OUTPUT_PATH = "D:\\stuff\\texts\\kaggle\\september2022\\results\\"

  def main(args: Array[String]): Unit = {

    println("September 2022 Kaggle competition")

    prepareEnv()

    processSeptember2022()
//    analyzeData()
  }

  private val extractYear: UserDefinedFunction = udf {
    (date: String) => {
      date.split("-")(0).toInt
    }
  }

  private val extractDayOfMonth: UserDefinedFunction = udf {
    (date: String) => {
      date.split("-")(2).toInt
    }
  }

  private val extractMonth: UserDefinedFunction = udf {
    (date: String) => {
      date.split("-")(1).toInt
    }
  }

  private val extractDayOfWeek: UserDefinedFunction = udf {
    (date: String) => {
      val arr = date.split("-")

      new DateTime(arr(0).toInt, arr(1).toInt, arr(2).toInt, 0, 0).dayOfWeek().get()
    }
  }

  private val extractIsEndOfYear: UserDefinedFunction = udf {
    (date: String) => {
      val arr = date.split("-")

      val d = new DateTime(arr(0).toInt, arr(1).toInt, arr(2).toInt, 0, 0)

      if (d.getMonthOfYear == 12 & d.getDayOfMonth >= 28 || d.getMonthOfYear == 1 & d.getDayOfMonth == 1) {
        true
      } else {
        false
      }
    }
  }

  private def processSeptember2022(): Unit = {

    val session = createSparkSession("kaggleSeptember2022")

    val df = createDf("D:\\stuff\\texts\\kaggle\\september2022\\train.csv", session)

    println(s"df count ${df.count()}")

    val countryIndexer = new StringIndexer()
      .setInputCol("country")
      .setOutputCol("indexedCountry")

    val storeIndexer = new StringIndexer()
      .setInputCol("store")
      .setOutputCol("indexedStore")

    val productIndexer = new StringIndexer()
      .setInputCol("product")
      .setOutputCol("indexedProduct")

    val countryOneHotEncoder = new OneHotEncoder()
      .setInputCol("indexedCountry")
      .setOutputCol("hotCountry")

    val storeOneHotEncoder = new OneHotEncoder()
      .setInputCol("indexedStore")
      .setOutputCol("hotStore")

    val productOneHotEncoder = new OneHotEncoder()
      .setInputCol("indexedProduct")
      .setOutputCol("hotProduct")

    val assembler = new VectorAssembler()
      .setInputCols(Array(
//        "hotCountry",
        "hotStore",
        "hotProduct",
        "dayOfWeek",
        "month",
        "dayOfMonth",
        "isNearHoliday")
      )
      .setOutputCol("features")

    val regressor = new RandomForestRegressor().setLabelCol("num_sold")


    val pipeline = new Pipeline().setStages(Array(countryIndexer, storeIndexer, productIndexer, countryOneHotEncoder, storeOneHotEncoder, productOneHotEncoder, assembler, regressor))

    val evaluator = new RegressionEvaluator()
      .setLabelCol("num_sold")

    val paramGrid = new ParamGridBuilder()
      .addGrid(regressor.maxBins, Array(60))
      .addGrid(regressor.maxDepth, Array(30))
      .addGrid(regressor.numTrees, Array(30))
      .build()

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)

    val Array(dfData, dfTestData) = df.randomSplit(Array(0.9, 0.1))

    val model = crossValidator.fit(dfData)

    val prediction = model.transform(dfTestData)

    val accuracy = evaluator.evaluate(prediction)

    println(s"accuracy $accuracy")

    val testDf = createDf("D:\\stuff\\texts\\kaggle\\september2022\\test.csv", session)

    val actualPrediction = model.transform(testDf)

    actualPrediction
      .withColumn("num_sold", col("prediction"))
      .select("row_id", "num_sold")
      .coalesce(1)
      .write.format("csv")
      .option("header", "true")
      .save(OUTPUT_PATH + "res-" + accuracy)

    println("### ok, all done")
  }

  private def createDf(path: String, session: SparkSession): DataFrame = {

    session.read.format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load(path)
      .withColumn("dayOfWeek", extractDayOfWeek(col("date")))
      .withColumn("month", extractMonth(col("date")))
      .withColumn("dayOfMonth", extractDayOfMonth(col("date")))
      .withColumn("isNearHoliday", extractIsEndOfYear(col("date")))
      .withColumn("year", extractYear(col("date")))
  }

  private def analyzeData(): Unit = {

    val session = createSparkSession("kaggleSeptember2022")

    val df = createDf("D:\\stuff\\texts\\kaggle\\september2022\\train.csv", session)


    println(s"df count ${df.count()}")

    implicit val outputEncoder: Encoder[String] = ExpressionEncoder()


//    printRowsCountByValue("country", df)
//
//    printRowsCountByValue("store", df)
//
//    printRowsCountByValue("product", df)


//    df.orderBy(desc("num_sold")).show(100)

//    val targets = df.select("country").distinct().map(r => r.getString(0)).collect()
//
//    println(targets.mkString(", "))
//
//    df.select("store").distinct().show()

//    df.groupBy("year", "month").sum("num_sold").orderBy("year", "month")
//      .coalesce(1)
//      .write.format("csv")
//      .option("header", "true")
//      .save(OUTPUT_PATH + "salesByMonthAndDate.csv")

    println(df.columns.mkString(", "))


  }
}
