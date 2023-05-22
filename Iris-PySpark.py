from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Iris Dataset") \
    .getOrCreate()

# Download the iris dataset CSV file and save it locally
iris_url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
local_path = "iris.csv"
import urllib.request
urllib.request.urlretrieve(iris_url, local_path)

# Load the iris dataset from the local file
irisData = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(local_path)

# Convert the "species" column from string to numeric
labelIndexer = StringIndexer(inputCol="species", outputCol="label")
irisData = labelIndexer.fit(irisData).transform(irisData)

# Prepare the data
featureColumns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
assembler = VectorAssembler(inputCols=featureColumns, outputCol="features")
assembledData = assembler.transform(irisData)

# Split the data into training and test sets
trainingData, testData = assembledData.randomSplit([0.7, 0.3])

# Create a Logistic Regression model
logisticRegression = LogisticRegression(labelCol="label", featuresCol="features")

# Train the model
model = logisticRegression.fit(trainingData)

# Make predictions on the test set
predictions = model.transform(testData)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Show the features and corresponding predictions
predictions.select("features", "prediction").show(truncate=False)

# Display the accuracy of the model
print(f"Accuracy: {accuracy * 100}%")

# Stop the SparkSession
spark.stop()
