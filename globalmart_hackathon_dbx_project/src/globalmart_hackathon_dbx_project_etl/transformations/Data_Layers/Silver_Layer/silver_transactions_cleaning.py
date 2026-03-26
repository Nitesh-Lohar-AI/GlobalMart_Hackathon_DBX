# =============================================================================
# GLOBALMART — SILVER LAYER TRANSACTIONS TRANSFORMATION
# Purpose: Standardize sales metrics, recover schema-drifted data from 
#          _rescued_data, and normalize payment/discount attributes.
# =============================================================================

from pyspark import pipelines as dp
from pyspark.sql.functions import *

# Configuration
catalog = "dbx_hck_glbl_mart"
read_schema = "bronze"
write_schema = "silver"

# -----------------------------
# 1. Data Quality Expectations
# -----------------------------
rules = {
    "valid_order_id": "order_id IS NOT NULL",
    "valid_product_id": "product_id IS NOT NULL",
    "valid_sales": "sales IS NOT NULL",
    "valid_payment_type": "payment_type IN ('credit_card','debit_card','voucher')"
}

# -----------------------------
# 2. Helper Functions
# -----------------------------

def clean_sales(column):
    """
    Removes currency symbols and casts sales values to double precision.
    """
    return regexp_replace(col(column), "\\$", "").cast("double")

def normalize_discount(column):
    """
    Standardizes discount formats by converting percentage strings 
    to numeric values.
    """
    return when(col(column).like("%\\%%"),
                regexp_replace(col(column), "%", "").cast("double")
           ).otherwise(col(column) * 100)

def fix_payment_type(column):
    """
    Imputes missing or invalid payment types with a default 'credit_card' value.
    """
    return when((col(column).isNull()) | (col(column) == "?"), lit("credit_card")) \
           .otherwise(col(column))

def fix_payment_installments(column):
    """
    Handles nulls and invalid strings in installment data, defaulting to 0.
    """
    return when(
        (col(column).isNull()) | 
        (col(column) == "?") | 
        (col(column) == "NULL"),
        lit(0)
    ).otherwise(col(column).cast("int"))

def recover_from_rescue(main_col, rescue_key):
    """
    Attempts to retrieve data from the primary column; if null, fetches 
    from the _rescued_data JSON blob (handles schema drift/casing issues).
    """
    return coalesce(col(main_col), get_json_object(col("_rescued_data"), f"$.{rescue_key}"))

# -----------------------------
# 3. Standardization Logic
# -----------------------------

@dp.temporary_view()
def silver_transactions_standardized():
    """
    Applies business logic to raw transactions, including data recovery 
    from JSON fields and multi-column cleaning.
    """
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_transactions")

    df = (
        df
        # Use helper to recover IDs that may have shifted during ingestion
        .withColumn("order_id", recover_from_rescue("Order_ID", "Order_id"))
        .withColumn("product_id", recover_from_rescue("Product_ID", "Product_id"))

        # Clean numeric sales data
        .withColumn("sales", clean_sales("Sales"))

        # Manual recovery for quantity to handle type casting from JSON
        .withColumn(
            "quantity",
            coalesce(
                col("Quantity"),
                get_json_object(col("_rescued_data"), "$.Quantity").cast("int")
            )
        )

        # Normalize financial metrics
        .withColumn("discount", normalize_discount("discount"))
        .withColumn("profit", coalesce(col("profit"), lit(0.0)))

        # Standardize payment information
        .withColumn("payment_type", fix_payment_type("payment_type"))
        .withColumn("payment_installments", fix_payment_installments("payment_installments"))

        # Rename system metadata columns for cleaner silver schema
        .withColumnRenamed("_source_file", "source_file")
        .withColumnRenamed("_load_timestamp", "load_timestamp")
        .withColumnRenamed("_region", "region")
    )

    return df

# -----------------------------
# 4. Final Validated Table
# -----------------------------

@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.{write_schema}.silver_transactions"
)
def silver_transactions_clean():
    """
    Primary Silver table for transactions. Enforces strict data quality 
    rules and drops non-compliant records.
    """
    return spark.read.table("silver_transactions_standardized")

# -----------------------------
# 5. Data Quality Quarantine
# -----------------------------

@dp.table(name=f"{catalog}.{write_schema}.silver_transactions_quarantine")
def silver_transactions_rejects():
    """
    Captures records failing transaction expectations. Dynamically lists 
    the specific rule violations for each record.
    """
    df = spark.read.table("silver_transactions_standardized")

    # Generate a list of failed rule names
    error_conditions = [
        when(~expr(rule), lit(rule_name))
        for rule_name, rule in rules.items()
    ]

    # Combine errors into a readable string
    df_with_errors = df.withColumn(
        "error_reason",
        concat_ws(",", *error_conditions)
    )

    return (
        df_with_errors
        .filter(col("error_reason") != "")
        .withColumn("quarantine_ts", current_timestamp())
    )