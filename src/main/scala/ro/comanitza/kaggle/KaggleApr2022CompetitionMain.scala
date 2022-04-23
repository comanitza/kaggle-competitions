package ro.comanitza.kaggle

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.{col, corr}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 *
 * Solution for the KAggle April 2022 challenge
 *
 * https://www.kaggle.com/competitions/tabular-playground-series-apr-2022
 *
 */
object KaggleApr2022CompetitionMain extends SparkBase {

  private val TRAIN_DATA_PATH = "D:\\stuff\\texts\\kaggle\\apr2022\\train.csv"
  private val LABEL_DATA_PATH = "D:\\stuff\\texts\\kaggle\\apr2022\\train_labels.csv"
  private val TEST_DATA_PATH = "D:\\stuff\\texts\\kaggle\\apr2022\\test.csv"
  private val OUTPUT_PATH = "D:\\stuff\\texts\\kaggle\\apr2022\\results\\"

  def main(args: Array[String]): Unit = {

    println("Kaggle Apr 2022 competition")

    prepareEnv()

    /*
     * determine the best columns to use, can be run just for tests
     */
    //determineBestCorrelations()

    /*
     * provide the solution and output int in the kagle file format
     */
    testApr2022Challenge()
  }

  /**
   * method for determining the best correlations between columns and labels
   */
  private def determineBestCorrelations(): Unit = {

    val session = createSparkSession()

    val df = session.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(TRAIN_DATA_PATH)


    val labelsDf = session.read.format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load(LABEL_DATA_PATH)
      .withColumn("labelSequence", col("sequence"))
      .drop("sequence")

    val joinedDf = df.join(labelsDf, df("sequence") === labelsDf("labelSequence") , "left")
      .withColumn("label", col("state"))
      .drop("state")

    println("joinedDf count: " + joinedDf.count())

    joinedDf.select(corr("sensor_00", "label")).show()
    joinedDf.select(corr("sensor_01", "label")).show()
    joinedDf.select(corr("sensor_02", "label")).show()
    joinedDf.select(corr("sensor_03", "label")).show()

    joinedDf.select(corr("sensor_04", "label")).show()
    joinedDf.select(corr("sensor_05", "label")).show()
    joinedDf.select(corr("sensor_06", "label")).show()
    joinedDf.select(corr("sensor_07", "label")).show()
    joinedDf.select(corr("sensor_08", "label")).show()

    joinedDf.select(corr("sensor_09", "label")).show()
    joinedDf.select(corr("sensor_10", "label")).show()
    joinedDf.select(corr("sensor_11", "label")).show()
    joinedDf.select(corr("sensor_12", "label")).show()
  }

  private def loadData(path: String, session: SparkSession = createSparkSession()): DataFrame = {

    val rdd = session.sparkContext
      .textFile(path)
      .filter(s => !s.contains("sequence"))
      .map(parseSequenceDto)

    val composedDto = rdd.groupBy(dto => dto.getSequence).map(t => {

      val asList = t._2.toList

      val sensor00Arr = new Array[Double](asList.size)
      val sensor01Arr = new Array[Double](asList.size)
      val sensor02Arr = new Array[Double](asList.size)
      val sensor03Arr = new Array[Double](asList.size)
      val sensor04Arr = new Array[Double](asList.size)
      val sensor05Arr = new Array[Double](asList.size)
      val sensor06Arr = new Array[Double](asList.size)
      val sensor07Arr = new Array[Double](asList.size)
      val sensor08Arr = new Array[Double](asList.size)
      val sensor09Arr = new Array[Double](asList.size)
      val sensor10Arr = new Array[Double](asList.size)
      val sensor11Arr = new Array[Double](asList.size)
      val sensor12Arr = new Array[Double](asList.size)

      for(i <- asList.indices) {

        sensor00Arr(i) = asList(i).sensor00
        sensor01Arr(i) = asList(i).sensor01
        sensor02Arr(i) = asList(i).sensor02
        sensor03Arr(i) = asList(i).sensor03
        sensor04Arr(i) = asList(i).sensor04
        sensor05Arr(i) = asList(i).sensor05
        sensor06Arr(i) = asList(i).sensor06
        sensor07Arr(i) = asList(i).sensor07
        sensor08Arr(i) = asList(i).sensor08
        sensor09Arr(i) = asList(i).sensor09
        sensor10Arr(i) = asList(i).sensor10
        sensor11Arr(i) = asList(i).sensor11
        sensor12Arr(i) = asList(i).sensor12
      }

      new CompoundedSequenceDto(asList.head.getSequence,
        asList.head.getSubject,
        asList.head.getStep,
        Vectors.dense(sensor00Arr),
        Vectors.dense(sensor01Arr),
        Vectors.dense(sensor02Arr),
        Vectors.dense(sensor03Arr),
        Vectors.dense(sensor04Arr),
        Vectors.dense(sensor05Arr),
        Vectors.dense(sensor06Arr),
        Vectors.dense(sensor07Arr),
        Vectors.dense(sensor08Arr),
        Vectors.dense(sensor09Arr),
        Vectors.dense(sensor10Arr),
        Vectors.dense(sensor11Arr),
        Vectors.dense(sensor12Arr))
    })

    val df = session.createDataFrame(composedDto, classOf[CompoundedSequenceDto])

    val labelsDf = session.read.format("csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load(LABEL_DATA_PATH)
      .withColumn("labelSequence", col("sequence"))
      .drop("sequence")

    val joinedDf = df.join(labelsDf, df("sequence") === labelsDf("labelSequence") , "left")
      .withColumn("label", col("state"))
      .drop("state")

    joinedDf
  }

  private def testApr2022Challenge(): Unit = {
    val session = createSparkSession()

    val joinedDf = loadData(TRAIN_DATA_PATH, session)

    val featuresArr = Array(
      "sensor02",
      "sensor04",
      "sensor07",
      "sensor08",
      "sensor10",
      "sensor12"
    )
    val assembler = new VectorAssembler()
      .setInputCols(featuresArr)
      .setOutputCol("features")

    val classifier = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(assembler, classifier))

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(28))
      .addGrid(classifier.maxDepth, Array(28))
      .addGrid(classifier.numTrees, Array(28))
      .build()

    val crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)

    val Array(trainDataDf, testDataDf) = joinedDf.randomSplit(Array(0.9, 0.1))

    val model = crossValidator.fit(trainDataDf)

    val prediction = model.transform(testDataDf)

    val accuracy = evaluator.evaluate(prediction)

    println("accuracy: " + accuracy)

    val testDf = loadData(TEST_DATA_PATH)

    println("test size: " + testDf.count())

    val actualPrediction = model.transform(testDf)

    actualPrediction.withColumn("state", col("prediction"))
      .select("sequence", "state")
      .coalesce(1)
      .write.format("csv")
      .option("header", "true")
      .save(OUTPUT_PATH + "res-" + accuracy + "-" +System.currentTimeMillis())

    println("### ok, all done")
  }

  private def parseSequenceDto(line: String): SequenceDto = {

    val arr = line.split(",")

    new SequenceDto(arr(0).toInt,
      arr(1).toInt,
      arr(2).toInt,
      arr(3).toDouble,
      arr(4).toDouble,
      arr(5).toDouble,
      arr(6).toDouble,
      arr(7).toDouble,
      arr(8).toDouble,
      arr(9).toDouble,
      arr(10).toDouble,
      arr(11).toDouble,
      arr(12).toDouble,
      arr(13).toDouble,
      arr(14).toDouble,
      arr(15).toDouble)
  }
}

class SequenceDto(private val sequence: Int,
                  private val subject: Int,
                  private val step: Int,
                  val sensor00: Double,
                  val sensor01: Double,
                  val sensor02: Double,
                  val sensor03: Double,
                  val sensor04: Double,
                  val sensor05: Double,
                  val sensor06: Double,
                  val sensor07: Double,
                  val sensor08: Double,
                  val sensor09: Double,
                  val sensor10: Double,
                  val sensor11: Double,
                  val sensor12: Double) extends Serializable {

  def getSequence: Int = sequence
  def getSubject: Int = subject
  def getStep: Int = step

  override def toString = s"SequenceDto(sequence=$sequence, subject=$subject, step=$step, sensor00=$sensor00, sensor01=$sensor01, sensor02=$sensor02, sensor03=$sensor03, sensor04=$sensor04, sensor05=$sensor05, sensor06=$sensor06, sensor07=$sensor07, sensor08=$sensor08, sensor09=$sensor09, sensor10=$sensor10, sensor11=$sensor11, sensor12=$sensor12)"
}

class CompoundedSequenceDto(private val sequence: Int,
                            private val subject: Int,
                            private val step: Int,
                            private val sensor00: Vector,
                            private val sensor01: Vector,
                            private val sensor02: Vector,
                            private val sensor03: Vector,
                            private val sensor04: Vector,
                            private val sensor05: Vector,
                            private val sensor06: Vector,
                            private val sensor07: Vector,
                            private val sensor08: Vector,
                            private val sensor09: Vector,
                            private val sensor10: Vector,
                            private val sensor11: Vector,
                            private val sensor12: Vector) extends Serializable {

  def getSequence: Int = sequence
  def getSubject: Int = subject
  def getStep: Int = step

  def getSensor00: Vector = sensor00
  def getSensor01: Vector = sensor01
  def getSensor02: Vector = sensor02
  def getSensor03: Vector = sensor03
  def getSensor04: Vector = sensor04
  def getSensor05: Vector = sensor05
  def getSensor06: Vector = sensor06
  def getSensor07: Vector = sensor07
  def getSensor08: Vector = sensor08
  def getSensor09: Vector = sensor09
  def getSensor10: Vector = sensor10
  def getSensor11: Vector = sensor11
  def getSensor12: Vector = sensor12
}
