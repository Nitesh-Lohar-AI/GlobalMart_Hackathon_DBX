# =============================================================================
# GLOBALMART — SILVER LAYER RETURNS TRANSFORMATION
# Purpose: Cleanse and standardize return transactions, handling multiple date
#          formats and inconsistent status codes across regions.
# =============================================================================

from pyspark import pipelines as dp
from pyspark.sql.functions import col, coalesce, when, regexp_replace, concat_ws, trim, expr, lit, current_timestamp, to_date

# Configuration
catalog = 'dbx_hck_glbl_mart'
read_schema = 'bronze'
table = 'bronze_returns'

# Data Quality Expectations for Returns
rules = {
    "valid_order_id": "order_id IS NOT NULL",
    "valid_return_reason": "return_reason IS NOT NULL AND return_reason != ''",
    "valid_return_status": "return_status IN ('Pending', 'Rejected', 'Approved')",
    "valid_refund_amount": "refund_amount IS NOT NULL"
}

# -----------------------------
# 1. Helper Functions
# -----------------------------

def clean_amount(column):
    """
    Removes currency symbols and casts the refund amount to a double.
    """
    return regexp_replace(column, "\\$", "").cast("double")

def clean_reason(column):
    """
    Removes noise characters from the return reason text.
    """
    return regexp_replace(column, "\\?", "").cast("string")    

def normalize_status(column):
    """
    Maps abbreviated status codes to standardized business terms.
    """
    return when(column == "APPRVD", "Approved") \
        .when(column == "RJCTD", "Rejected") \
        .when(column == "PENDG", "Pending") \
        .otherwise(column)

# -----------------------------
# 2. Standardization Logic
# -----------------------------

@dp.temporary_view()
def silver_returns_standardized():
    """
    Harmonizes return data by coalescing regional column variants, 
    fixing date formats, and cleaning numeric/text fields.
    """
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_returns")

    # Coalesce differing column names from various source systems
    df1 = df.select(
        coalesce(col("OrderId"), col("order_id")).alias("order_id"),
        coalesce(col("amount"), col("refund_amount")).alias("refund_amount"),
        coalesce(col("date_of_return"), col("return_date")).alias("return_date"),
        coalesce(col("status"), col("return_status")).alias("return_status"),
        coalesce(col("reason"), col("return_reason")).alias("return_reason"),
        col("_rescued_data"),
        col("_load_timestamp").alias("load_timestamp"),
        col("_region").alias("region")
    )

    df2 = (
        df1
        # Apply status normalization
        .withColumn("return_status", normalize_status(col("return_status")))

        # Remove invalid characters from reason
        .withColumn("return_reason", clean_reason(col("return_reason")))

        # Handle explicit string "NULL" values in date fields
        .withColumn(
            "return_date",
            when(col("return_date") == "NULL", None).otherwise(col("return_date"))
        )

        # Multi-format date parsing (ISO and US formats)
        .withColumn(
            "return_date",
            when(
                col("return_date").rlike(r"^\d{4}-\d{2}-\d{2}$"),
                to_date(col("return_date"), "yyyy-MM-dd")
            ).when(
                col("return_date").rlike(r"^\d{2}-\d{2}-\d{4}$"),
                to_date(col("return_date"), "MM-dd-yyyy")
        ).otherwise(None))

        # Clean currency characters and whitespace
        .withColumn("refund_amount", clean_amount(col("refund_amount")))
        .withColumn("return_reason", trim(col("return_reason")))
    )

    return df2

# -----------------------------
# 3. Final Validated Table
# -----------------------------

@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.silver.silver_returns"
)
def silver_returns_clean():
    """
    Final Silver table containing only validated return records.
    """
    return spark.read.table("silver_returns_standardized")

# -----------------------------
# 4. Data Quality Quarantine
# -----------------------------

@dp.table(name=f"{catalog}.silver.silver_returns_quarantine")
def silver_returns_rejects():
    """
    Captures records failing return-specific expectations for audit.
    """
    df = spark.read.table("silver_returns_standardized")

    # Evaluate all quality conditions
    error_conditions = [
        when(~expr(rule), lit(rule_name))
        for rule_name, rule in rules.items()
    ]

    # Combine failed rules into a single audit column
    df_with_errors = df.withColumn(
        "error_reason",
        concat_ws(",", *error_conditions)
    )

    return (
        df_with_errors
        .filter(col("error_reason") != "")
        .withColumn("quarantine_ts", current_timestamp())
    )