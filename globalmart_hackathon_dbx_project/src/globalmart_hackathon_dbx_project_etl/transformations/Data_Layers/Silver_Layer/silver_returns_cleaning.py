from pyspark import pipelines as dp
from pyspark.sql.functions import col, coalesce, when, regexp_replace, concat_ws, trim, expr, lit,current_timestamp, to_date

catalog = 'dbx_hck_glbl_mart'
read_schema = 'bronze'
table = 'bronze_returns'

rules = {
    "valid_order_id": "order_id IS NOT NULL",
    "valid_return_reason": "return_reason IS NOT NULL AND return_reason != ''",
    "valid_return_status": "return_status IN ('Pending', 'Rejected', 'Approved')",
    "valid_refund_amount": "refund_amount IS NOT NULL"
}

def clean_amount(column):
    return regexp_replace(column, "\\$", "").cast("double")

def clean_reason(column):
    return regexp_replace(column, "\\?", "").cast("string")    

def normalize_status(column):
    return when(column == "APPRVD", "Approved") \
        .when(column == "RJCTD", "Rejected") \
        .when(column == "PENDG", "Pending") \
        .otherwise(column)

@dp.temporary_view()
def silver_returns_standardized():

    df = spark.read.table(f"{catalog}.{read_schema}.bronze_returns")

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
        # Normalize status
        .withColumn("return_status", normalize_status(col("return_status")))

        # Clean reason
        .withColumn("return_reason", clean_reason(col("return_reason")))

        # Clean return date
        .withColumn(
            "return_date",
            when(col("return_date") == "NULL", None).otherwise(col("return_date"))
        )

        .withColumn(
            "return_date",
            when(
                col("return_date").rlike(r"^\d{4}-\d{2}-\d{2}$"),
                to_date(col("return_date"), "yyyy-MM-dd")
            ).when(
                col("return_date").rlike(r"^\d{2}-\d{2}-\d{4}$"),
                to_date(col("return_date"), "MM-dd-yyyy")
        ).otherwise(None))

        # Clean amount
        .withColumn("refund_amount", clean_amount(col("refund_amount")))

        # Trim text fields
        .withColumn("return_reason", trim(col("return_reason")))
    )

    return df2

@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.silver.silver_returns"
)
def silver_returns_clean():
    return spark.read.table("silver_returns_standardized")

@dp.table(name=f"{catalog}.silver.silver_returns_quarantine")
def silver_returns_rejects():
    df = spark.read.table("silver_returns_standardized")

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