from pyspark import pipelines as dp
from pyspark.sql.functions import *

catalog = 'dbx_hck_glbl_mart'
read_schema = 'bronze'
write_schema = 'silver'

from pyspark.sql.functions import col, when, to_timestamp, expr, lit, concat_ws, current_timestamp
from pyspark import pipelines as dp

# Define rules
rules = {
    "valid_order_id": "order_id IS NOT NULL",
    "valid_customer_id": "customer_id IS NOT NULL",
    "valid_vendor_id": "vendor_id IS NOT NULL",
    "valid_ship_mode": "ship_mode IN ('First Class','Second Class','Standard Class','Same Day')",
    "valid_order_purchase_date": "order_purchase_date IS NOT NULL", 
    "valid_order_status": "order_status IN ('delivered','shipped','unavailable','canceled','processing','invoiced','created')"
}


def parse_std_datetime(column):
    """
    Convert a column to timestamp type with standard format yyyy-MM-dd HH:mm
    """
    return to_timestamp(col(column), "yyyy-MM-dd HH:mm")

@dp.temporary_view()
def silver_orders_standardized():
    # 3a. Read standardized raw orders table
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_orders")
    
    # 3b. Standardize datetime columns
    datetime_cols = [
        "order_purchase_date",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    
    for col_name in datetime_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, parse_std_datetime(col_name))
        else:
            # Handle missing column (e.g., R5 issue)
            from pyspark.sql.functions import lit
            df = df.withColumn(col_name, lit(None).cast("timestamp"))
    
    # 3c. Standardize ship_mode
    df = df.withColumn(
        "ship_mode",
        when(col("ship_mode").isin("1st Class", "First Class"), "First Class")
        .when(col("ship_mode").isin("2nd Class", "Second Class"), "Second Class")
        .when(col("ship_mode").isin("Std Class", "Standard Class"), "Standard Class")
        .otherwise(col("ship_mode"))
    )
    
    # 3d. Return cleaned DataFrame
    return df


# 3️3 Orders table decorator
@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.{write_schema}.silver_orders_clean"
)
def silver_orders_clean():
    return spark.read.table("silver_orders_standardized")


@dp.table(name=f"{catalog}.{write_schema}.silver_orders_quaruntine")
def silver_orders_rejects():
    df = spark.read.table("bronze_orders")

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