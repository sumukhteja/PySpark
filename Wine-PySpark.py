import urllib.request
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Wine Dataset") \
    .getOrCreate()

# Define the schema of the dataset
schema = StructType([
    StructField("label", IntegerType(), nullable=True),
    StructField("features", DoubleType(), nullable=True),
])

# Download the Wine dataset from the UCI Machine Learning Repository
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
data_path = r"C:\Users\Sunny\wine.data"
urllib.request.urlretrieve(data_url, data_path)

# Read the dataset from the local file using the defined schema
wineData = spark.read.format("csv") \
    .schema(schema) \
    .option("header", "false") \
    .load(data_path)

# Prepare the data
assembler = VectorAssembler(inputCols=wineData.columns[1:], outputCol="transformed_features")
wineData = assembler.transform(wineData)

# Split the data into training and test sets
(trainingData, testData) = wineData.randomSplit([0.7, 0.3])

# Create a Random Forest Classifier
randomForest = RandomForestClassifier(labelCol="label", featuresCol="transformed_features")

# Train the model
model = randomForest.fit(trainingData)

# Make predictions on the test set
predictions = model.transform(testData)

# Show the predicted labels and corresponding features
predictions.select("label", "transformed_features", "prediction").show()

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy: {:.2%}".format(accuracy))
