# Import SparkSession to create and control the Spark application
from pyspark.sql import SparkSession

# Import required Spark SQL functions
from pyspark.sql.functions import col, trim


# ------------------------------------------------------------------
# STEP 1: Create a Spark Session
# ------------------------------------------------------------------
# SparkSession is the entry point to using Spark.
# It allows us to read data, run transformations, and manage jobs.
# Here we give the application a descriptive name.
spark = SparkSession.builder \
    .appName("TelcoCustomerChurn_Preprocess") \
    .getOrCreate()


# ------------------------------------------------------------------
# STEP 2: Define Input CSV File Path
# ------------------------------------------------------------------
# This is the location of the raw dataset on the local machine.
# The dataset contains customer information and churn labels.
input_csv = r"C:\Users\Lenovo\Desktop\archive\WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ------------------------------------------------------------------
# STEP 3: Read the CSV File into a Spark DataFrame
# ------------------------------------------------------------------
# - header=True tells Spark that the first row contains column names
# - inferSchema=True allows Spark to automatically detect data types
#   (e.g., Integer, Double, String)
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(input_csv)
)


# ------------------------------------------------------------------
# STEP 4: Clean String Columns (Trim Extra Spaces)
# ------------------------------------------------------------------
# Some categorical values may contain leading or trailing spaces
# (e.g., " Yes", "Yes ").
# This loop applies the trim() function to all columns of type String
# to ensure consistency and avoid incorrect category duplication.
for column_name, column_type in df.dtypes:
    if column_type == "string":
        df = df.withColumn(column_name, trim(col(column_name)))


# ------------------------------------------------------------------
# STEP 5: Handle Missing Target Values
# ------------------------------------------------------------------
# The target column "Churn" is essential for supervised learning.
# Rows without a Churn value cannot be used for training,
# so they are removed from the dataset.
df = df.na.drop(subset=["Churn"])


# ------------------------------------------------------------------
# STEP 6: Handle Remaining Missing Values
# ------------------------------------------------------------------
# Any remaining missing values in other columns are filled
# with the value "Unknown".
# This prevents errors during model training and ensures
# a complete dataset.
df = df.na.fill("Unknown")


# ------------------------------------------------------------------
# STEP 7: Define Output Directory for Cleaned Data
# ------------------------------------------------------------------
# This directory will store the cleaned dataset after preprocessing.
# Spark normally writes data in multiple partitions.
output_dir = r"C:\Users\Lenovo\Desktop\archive\spark_output"


# ------------------------------------------------------------------
# STEP 8: Export the Cleaned Dataset
# ------------------------------------------------------------------
# - coalesce(1) is used to combine all partitions into a single CSV file
# - This is important because tools like Orange require one CSV file
# - mode("overwrite") allows replacing any existing output
# - header=True keeps column names in the exported file
(
    df.coalesce(1)
      .write
      .mode("overwrite")
      .option("header", True)
      .csv(output_dir)
)


# ------------------------------------------------------------------
# STEP 9: Stop the Spark Session
# ------------------------------------------------------------------
# Stops the Spark application and releases system resources.
spark.stop()