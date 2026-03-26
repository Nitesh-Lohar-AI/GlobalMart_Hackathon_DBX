from pyspark import pipelines as dp
from pyspark.sql.functions import *


catalog = "dbx_hck_glbl_mart"
read_schema = "bronze"
write_schema = "silver"

# -----------------------------
# 1. Define expectations / rules
# -----------------------------
rules = {
    "valid_order_id": "order_id IS NOT NULL",
    "valid_product_id": "product_id IS NOT NULL",
    "valid_sales": "sales IS NOT NULL",
    "valid_payment_type": "payment_type IN ('credit_card','debit_card','voucher')"
}

# -----------------------------
# 2. Helper functions
# -----------------------------

def clean_sales(column):
    return regexp_replace(col(column), "\\$", "").cast("double")

def normalize_discount(column):
    return when(col(column).like("%\\%%"),
                regexp_replace(col(column), "%", "").cast("double")
           ).otherwise(col(column) * 100)

def fix_payment_type(column):
    return when((col(column).isNull()) | (col(column) == "?"), lit("credit_card")) \
           .otherwise(col(column))

def fix_payment_installments(column):
    return when(
        (col(column).isNull()) | 
        (col(column) == "?") | 
        (col(column) == "NULL"),
        lit(0)
    ).otherwise(col(column).cast("int"))

def recover_from_rescue(main_col, rescue_key):
    return coalesce(col(main_col), get_json_object(col("_rescued_data"), f"$.{rescue_key}"))

# -----------------------------
# 3. Temporary view: standardize transactions
# -----------------------------
@dp.temporary_view()
def silver_transactions_standardized():
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_transactions")

    df = (
        df
        # Recover keys
        .withColumn("order_id", recover_from_rescue("Order_ID", "Order_id"))
        .withColumn("product_id", recover_from_rescue("Product_ID", "Product_id"))

        # Clean sales
        .withColumn("sales", clean_sales("Sales"))

        # Quantity recovery
        .withColumn(
            "quantity",
            coalesce(
                col("Quantity"),
                get_json_object(col("_rescued_data"), "$.Quantity").cast("int")
            )
        )

        # Discount normalization
        .withColumn("discount", normalize_discount("discount"))

        # Profit handling
        .withColumn("profit", coalesce(col("profit"), lit(0.0)))

        # Payment cleanup
        .withColumn("payment_type", fix_payment_type("payment_type"))

        #payment_installments
        .withColumn("payment_installments", fix_payment_installments("payment_installments"))

        # Metadata rename
        .withColumnRenamed("_source_file", "source_file")
        .withColumnRenamed("_load_timestamp", "load_timestamp")
        .withColumnRenamed("_region", "region")
    )

    return df

# -----------------------------
# 4. Cleaned transactions table with expectations
# -----------------------------
@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.{write_schema}.silver_transactions"
)
def silver_transactions_clean():
    return spark.read.table("silver_transactions_standardized")

# -----------------------------
# 5. Quarantine table for rejected products
# -----------------------------
@dp.table(name=f"{catalog}.{write_schema}.silver_transactions_quarantine")
def silver_transactions_rejects():
    df = spark.read.table("silver_transactions_standardized")

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