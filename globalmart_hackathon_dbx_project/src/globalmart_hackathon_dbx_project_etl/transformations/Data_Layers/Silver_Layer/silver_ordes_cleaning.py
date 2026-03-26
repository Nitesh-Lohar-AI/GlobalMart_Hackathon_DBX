# Databricks notebook source
# =============================================================================
# GLOBALMART — SILVER LAYER ORDERS TRANSFORMATION
# Purpose: Clean, standardize, and validate order data including date parsing.
# =============================================================================

from pyspark.sql.functions import col, when, to_timestamp, expr, lit, concat_ws, current_timestamp
from pyspark import pipelines as dp

# Configuration
catalog = 'dbx_hck_glbl_mart'
read_schema = 'bronze'
write_schema = 'silver'

# Data Quality Rules for Expectations
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
    Converts a string column to a timestamp type using a consistent format.
    """
    return to_timestamp(col(column), "yyyy-MM-dd HH:mm")

@dp.temporary_view()
def silver_orders_standardized():
    """
    Standardizes raw order data by handling date conversion, filling missing 
    columns, and unifying shipping mode terminology.
    """
    # Load raw data from Bronze
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_orders")
    
    # Standardize all datetime-related columns
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
            # Handle regional missing columns (e.g., Region 5) by providing NULLs
            df = df.withColumn(col_name, lit(None).cast("timestamp"))
    
    # Normalize shipping mode variants to standard business terms
    df = df.withColumn(
        "ship_mode",
        when(col("ship_mode").isin("1st Class", "First Class"), "First Class")
        .when(col("ship_mode").isin("2nd Class", "Second Class"), "Second Class")
        .when(col("ship_mode").isin("Std Class", "Standard Class"), "Standard Class")
        .otherwise(col("ship_mode"))
    )
    
    return df

@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.{write_schema}.silver_orders_clean"
)
def silver_orders_clean():
    """
    Final Silver table for valid orders. Drops any records that do not 
    meet the defined data quality rules.
    """
    return spark.read.table("silver_orders_standardized")

@dp.table(name=f"{catalog}.{write_schema}.silver_orders_quaruntine")
def silver_orders_rejects():
    """
    Captures records failing validation into a quarantine table. 
    Concatenates specific rule violations into an error_reason column.
    """
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_orders")

    # Evaluate which specific rules were violated
    error_conditions = [
        when(~expr(rule), lit(rule_name))
        for rule_name, rule in rules.items()
    ]

    # Combine all violated rules into a comma-separated list
    df_with_errors = df.withColumn(
        "error_reason",
        concat_ws(",", *error_conditions)
    )

    return (
        df_with_errors
        .filter(col("error_reason") != "")
        .withColumn("quarantine_ts", current_timestamp())
    )