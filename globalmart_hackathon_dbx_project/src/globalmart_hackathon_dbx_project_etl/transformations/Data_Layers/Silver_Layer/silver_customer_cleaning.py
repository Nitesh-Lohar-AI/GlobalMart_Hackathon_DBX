# =============================================================================
# GLOBALMART — SILVER LAYER CUSTOMER TRANSFORMATION
# Purpose: Standardize, clean, and validate customer data from all regions.
# =============================================================================

from pyspark import pipelines as dp
from pyspark.sql.functions import *

# Configuration
catalog = 'dbx_hck_glbl_mart'
read_schema = 'bronze'
write_schema = 'silver'

@dp.temporary_view()
def silver_customers_standardized():
    """
    Standardizes raw bronze customer data by resolving schema variants, 
    normalizing segments/regions, and correcting regional data swaps.
    """
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_customers")

    return df.select(
        # Consolidate multiple ID variants into a single customer_id
        coalesce(col("customer_id"), col("CustomerID"), col("cust_id"), col("customer_identifier")).alias("customer_id"),

        # Unify email fields across different regional source schemas
        coalesce(col("customer_email"), col("email_address")).alias("customer_email"),

        # Unify name fields
        coalesce(col("customer_name"), col("full_name")).alias("customer_name"),

        # Normalize segment values to standard business categories
        when(lower(coalesce(col("segment"), col("customer_segment"))).isin("consumer","cons","cosumer"), "Consumer")
        .when(lower(coalesce(col("segment"), col("customer_segment"))).isin("corporate","corp"), "Corporate")
        .when(lower(coalesce(col("segment"), col("customer_segment"))).isin("home office","ho"), "Home Office")
        .otherwise(None).alias("segment"),

        col("country"),

        # Correct City/State swap specifically observed in Region 4 source files
        when(col("_region") == "region 4", col("state")).otherwise(col("city")).alias("city"),
        when(col("_region") == "region 4", col("city")).otherwise(col("state")).alias("state"),

        col("postal_code"),

        # Map short-hand region codes to full standardized names
        when(col("region").isin("E","East"), "East")
        .when(col("region").isin("W","West"), "West")
        .when(col("region").isin("S","South"), "South")
        .when(col("region").isin("N","North"), "North")
        .when(col("region") == "Central", "Central")
        .otherwise(None).alias("region"),

        # Maintain metadata for lineage and auditing
        col("_source_file"),
        col("_load_timestamp"),
        col("_region")
    )


@dp.table(name=f"{catalog}.{write_schema}.silver_customers_quaruntine")
def customers_rejects():
    """
    Identifies and captures records that fail data quality checks into a 
    quarantine table for troubleshooting and manual review.
    """
    df = spark.read.table("silver_customers_standardized")
    return (
        df.withColumn(
            "error_reason",
            when(col("customer_id").isNull(), "NULL_CUSTOMER_ID")
            .when(~col("segment").isin("Consumer","Corporate","Home Office"), "INVALID_SEGMENT")
            .when(~col("region").isin("East","West","South","North","Central"), "INVALID_REGION")
        )
        .filter(col("error_reason").isNotNull())
    )


# Define Expectations for DLT data quality enforcement
rules = {
    "valid_customer_id": "customer_id IS NOT NULL",
    "valid_segment": "segment IN ('Consumer','Corporate','Home Office')",
    "valid_region": "region IN ('East','West','South','North','Central')"
}

@dp.expect_all_or_drop(rules)
@dp.table(
    name=f"{catalog}.{write_schema}.silver_customers"
)
def customers_clean():
    """
    Final Silver table containing only validated customer records.
    Records failing the 'rules' expectations are dropped from this target.
    """
    return spark.read.table("silver_customers_standardized")