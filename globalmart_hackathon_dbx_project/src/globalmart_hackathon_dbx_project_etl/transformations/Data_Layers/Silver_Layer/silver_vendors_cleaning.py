from pyspark import pipelines as dp
from pyspark.sql.functions import *


catalog = "dbx_hck_glbl_mart"
read_schema = "bronze"
write_schema = "silver"

# -----------------------------
# 1. Define expectations / rules
# -----------------------------
rules = {
    "valid_vendor_id": "vendor_id IS NOT NULL",
    "valid_vendor_name": "vendor_name IS NOT NULL"
}

# -----------------------------
# 2. Cleaned products table with expectations
# -----------------------------
@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.{write_schema}.silver_vendors"
)
def silver_vendors_clean():
    return spark.read.table(f"{catalog}.{read_schema}.bronze_vendors")

# -----------------------------
# 5. Quarantine table for rejected products
# -----------------------------
@dp.table(name=f"{catalog}.{write_schema}.silver_vendors_quarantine")
def silver_vendors_rejects():
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_vendors")

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