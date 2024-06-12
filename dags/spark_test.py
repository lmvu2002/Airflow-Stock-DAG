from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("Example Spark Job") \
    .getOrCreate()

# Sample data to create a DataFrame
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
columns = ["name", "age"]

# Create a DataFrame from the sample data
df = spark.createDataFrame(data, columns)

# Perform some operations on the DataFrame (e.g., filter)
filtered_df = df.filter(df.age > 30)

# Write the filtered DataFrame to Parquet format
output_path = "/output"  # Replace with your HDFS path
filtered_df.write.mode("overwrite").parquet(output_path)

# Stop the Spark session
spark.stop()