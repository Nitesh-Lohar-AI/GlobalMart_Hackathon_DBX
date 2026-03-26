from pyspark import pipelines as dp
from pyspark.sql.functions import *

catalog = 'dbx_hck_glbl_mart'
read_schema = 'bronze'
write_schema = 'silver'

@dp.temporary_view()
def silver_customers_standardized():
    df = spark.read.table(f"{catalog}.{read_schema}.bronze_customers")

    return df.select(
        coalesce(col("customer_id"), col("CustomerID"), col("cust_id"), col("customer_identifier")).alias("customer_id"),

        coalesce(col("customer_email"), col("email_address")).alias("customer_email"),

        coalesce(col("customer_name"), col("full_name")).alias("customer_name"),

        # Segment normalization
        when(lower(coalesce(col("segment"), col("customer_segment"))).isin("consumer","cons","cosumer"), "Consumer")
        .when(lower(coalesce(col("segment"), col("customer_segment"))).isin("corporate","corp"), "Corporate")
        .when(lower(coalesce(col("segment"), col("customer_segment"))).isin("home office","ho"), "Home Office")
        .otherwise(None).alias("segment"),

        col("country"),

        # Fix region 4 issue
        when(col("_region") == "region 4", col("state")).otherwise(col("city")).alias("city"),
        when(col("_region") == "region 4", col("city")).otherwise(col("state")).alias("state"),

        col("postal_code"),

        # Region normalization
        when(col("region").isin("E","East"), "East")
        .when(col("region").isin("W","West"), "West")
        .when(col("region").isin("S","South"), "South")
        .when(col("region").isin("N","North"), "North")
        .when(col("region") == "Central", "Central")
        .otherwise(None).alias("region"),

        col("_source_file"),
        col("_load_timestamp"),
        col("_region")
    )


@dp.table(name=f"{catalog}.{write_schema}.silver_customers_quaruntine")
def customers_rejects():
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
    return spark.read.table("silver_customers_standardized")