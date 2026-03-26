# =============================================================================
# GLOBALMART — SILVER LAYER VENDOR TRANSFORMATION
# Purpose: Validate vendor master data and enforce identity integrity.
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
    "valid_vendor_id": "vendor_id IS NOT NULL",
    "valid_vendor_name": "vendor_name IS NOT NULL"
}

# -----------------------------
# 2. Final Validated Table
# -----------------------------
@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.{write_schema}.silver_vendors"
)
def silver_vendors_clean():
    """
    Primary Silver table for vendor master data. Enforces mandatory 
    identity fields and drops incomplete records.
    """
    return spark.read.table(f"{catalog}.{read_schema}.bronze_vendors")

# -----------------------------
# 3. Data Quality Quarantine
# -----------------------------
@dp.table(name=f"{catalog}.{write_schema}.silver_vendors_quarantine")
def silver_vendors_rejects():
    """
    Captures vendor records failing validation. Logs the specific rule 
    violation and a processing timestamp for auditing.
    """
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_vendors")

    # Check for violations against the defined rule set
    error_conditions = [
        when(~expr(rule), lit(rule_name))
        for rule_name, rule in rules.items()
    ]

    # Combine identified errors into a single string column
    df_with_errors = df.withColumn(
        "error_reason",
        concat_ws(",", *error_conditions)
    )

    return (
        df_with_errors
        .filter(col("error_reason") != "")
        .withColumn("quarantine_ts", current_timestamp())
    )