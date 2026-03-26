from pyspark import pipelines as dp
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, ArrayType, IntegerType


catalog = "dbx_hck_glbl_mart"
read_schema = "bronze"
write_schema = "silver"

# -----------------------------
# 1. Define expectations / rules
# -----------------------------
rules = {
    "valid_product_id": "product_id IS NOT NULL",
    "valid_product_name": "product_name IS NOT NULL",
    "valid_upc": "upc IS NOT NULL",
    "valid_product_photos_qty": "product_photos_qty >= 0"
}

# -----------------------------
# 2. Helper functions
# -----------------------------
def parse_std_datetime(column):
    """
    Convert a column to timestamp type with standard format yyyy-MM-dd HH:mm
    """
    return to_timestamp(col(column), "yyyy-MM-dd HH:mm")

def clean_upc(column):
    """
    Convert UPC from scientific notation to string without exponent
    """
    return when(col(column).isNotNull(),
                regexp_replace(col(column).cast("decimal(20,0)").cast(StringType()), "\.0$", "")
               ).otherwise(None)

# def split_sizes(column):
#     """
#     Split comma-separated sizes into array
#     """
#     return when(col(column).isNotNull(), split(col(column), ",")).otherwise(lit([]).cast(ArrayType(StringType())))

def normalize_categories(column):
    """
    Split categories string into array
    """
    return when(col(column).isNotNull(), split(col(column), ",")).otherwise(lit([]).cast(ArrayType(StringType())))

# -----------------------------
# 3. Temporary view: standardize products
# -----------------------------
@dp.temporary_view()
def silver_products_standardized():
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_products")

    # 3a. Standardize datetime columns
    datetime_cols = ["dateAdded", "dateUpdated"]
    for col_name in datetime_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, date_format(to_timestamp(col(col_name)), "yyyy-MM-dd HH:mm"))
        else:
            df = df.withColumn(col_name, lit(None).cast("timestamp"))

    # 3b. Normalize categories & sizes
    df = df.withColumn("categories_list", normalize_categories("categories"))
    # df = df.withColumn("sizes_list", split_sizes("sizes"))

    # 3c. Clean UPC
    df = df.withColumn("upc_cleaned", clean_upc("upc"))

    # 3d. Ensure product_photos_qty is integer and >=0
    df = df.withColumn("product_photos_qty", 
                       when(col("product_photos_qty").cast(IntegerType()).isNotNull(),
                            col("product_photos_qty").cast(IntegerType()))
                       .otherwise(lit(0))
                      )

    # 3e. Strip spaces from brand and product_name
    df = df.withColumn("brand", trim(col("brand"))) \
           .withColumn("product_name", trim(col("product_name")))

    return df

# -----------------------------
# 4. Cleaned products table with expectations
# -----------------------------
@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.{write_schema}.silver_products"
)
def silver_products_clean():
    return spark.read.table("silver_products_standardized")

# -----------------------------
# 5. Quarantine table for rejected products
# -----------------------------
@dp.table(name=f"{catalog}.{write_schema}.silver_products_quarantine")
def silver_products_rejects():
    df = spark.read.table(f"silver_products_standardized")

    error_conditions = [
        when(~expr(rule), lit(rule_name))
        for rule_name, rule in rules.items()
    ]

    df_with_errors = df.withColumn(
        "error_reason",
        concat_ws(",", *error_conditions)
    )

    return (
        df_with_errors
        .filter(col("error_reason") != "")
        .withColumn("quarantine_ts", current_timestamp())
    )