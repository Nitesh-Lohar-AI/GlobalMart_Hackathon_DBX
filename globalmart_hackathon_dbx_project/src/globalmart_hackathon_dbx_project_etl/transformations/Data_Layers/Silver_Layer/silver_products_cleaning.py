# =============================================================================
# GLOBALMART — SILVER LAYER PRODUCT TRANSFORMATION
# Purpose: Standardize product catalog, clean UPC codes, and validate schemas.
# =============================================================================

from pyspark import pipelines as dp
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, ArrayType, IntegerType

# Configuration
catalog = "dbx_hck_glbl_mart"
read_schema = "bronze"
write_schema = "silver"

# -----------------------------
# 1. Data Quality Expectations
# -----------------------------
rules = {
    "valid_product_id": "product_id IS NOT NULL",
    "valid_product_name": "product_name IS NOT NULL",
    "valid_upc": "upc IS NOT NULL",
    "valid_product_photos_qty": "product_photos_qty >= 0"
}

# -----------------------------
# 2. Helper Functions
# -----------------------------
def parse_std_datetime(column):
    """
    Converts string-based date columns into standardized timestamp objects.
    """
    return to_timestamp(col(column), "yyyy-MM-dd HH:mm")

def clean_upc(column):
    """
    Resolves scientific notation in UPC strings by casting through decimal 
    to preserve large numeric identifiers without exponents.
    """
    return when(col(column).isNotNull(),
                regexp_replace(col(column).cast("decimal(20,0)").cast(StringType()), "\.0$", "")
               ).otherwise(None)

def normalize_categories(column):
    """
    Transforms comma-separated category strings into structured arrays for 
    improved downstream filtering and search.
    """
    return when(col(column).isNotNull(), split(col(column), ",")).otherwise(lit([]).cast(ArrayType(StringType())))

# -----------------------------
# 3. Standardization Logic
# -----------------------------
@dp.temporary_view()
def silver_products_standardized():
    """
    Performs initial data cleaning including date formatting, UPC correction, 
    and attribute trimming before validation.
    """
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_products")

    # Standardize audit date columns
    datetime_cols = ["dateAdded", "dateUpdated"]
    for col_name in datetime_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, date_format(to_timestamp(col(col_name)), "yyyy-MM-dd HH:mm"))
        else:
            df = df.withColumn(col_name, lit(None).cast("timestamp"))

    # Convert flat category strings to arrays
    df = df.withColumn("categories_list", normalize_categories("categories"))

    # Fix scientific notation issues in UPC data
    df = df.withColumn("upc_cleaned", clean_upc("upc"))

    # Ensure photo quantities are valid integers, defaulting to 0 for nulls
    df = df.withColumn("product_photos_qty", 
                        when(col("product_photos_qty").cast(IntegerType()).isNotNull(),
                             col("product_photos_qty").cast(IntegerType()))
                        .otherwise(lit(0))
                       )

    # Remove leading/trailing white spaces from key text fields
    df = df.withColumn("brand", trim(col("brand"))) \
           .withColumn("product_name", trim(col("product_name")))

    return df

# -----------------------------
# 4. Final Validated Table
# -----------------------------
@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.{write_schema}.silver_products"
)
def silver_products_clean():
    """
    Primary Silver table for products. Enforces quality rules and 
    drops records that fail mandatory fields like product_id or upc.
    """
    return spark.read.table("silver_products_standardized")

# -----------------------------
# 5. Data Quality Quarantine
# -----------------------------
@dp.table(name=f"{catalog}.{write_schema}.silver_products_quarantine")
def silver_products_rejects():
    """
    Stores rejected product records. Provides a concatenated 'error_reason' 
    identifying which specific data quality rules were violated.
    """
    df = spark.read.table(f"silver_products_standardized")

    # Generate labels for failed conditions
    error_conditions = [
        when(~expr(rule), lit(rule_name))
        for rule_name, rule in rules.items()
    ]

    # Concatenate all specific error flags into one field
    df_with_errors = df.withColumn(
        "error_reason",
        concat_ws(",", *error_conditions)
    )

    return (
        df_with_errors
        .filter(col("error_reason") != "")
        .withColumn("quarantine_ts", current_timestamp())
    )