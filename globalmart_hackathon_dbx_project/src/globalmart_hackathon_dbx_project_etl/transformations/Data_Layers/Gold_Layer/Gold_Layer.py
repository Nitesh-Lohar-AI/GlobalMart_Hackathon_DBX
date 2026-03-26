# Databricks notebook source
# =============================================================================
# GLOBALMART — GOLD LAYER PIPELINE
# Spark Declarative Pipelines
# =============================================================================

from pyspark import pipelines as dp
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CATALOG = "dbx_hck_glbl_mart"
SILVER  = f"{CATALOG}.silver"
GOLD    = "gold"   # set as Target schema in Pipeline Settings UI

# COMMAND ----------

# ===========================================================================
# SECTION 1 — DIMENSION TABLES
# FIX: Dimensions use @dp.materialized_view() decorator
#      because silver sources are Delta batch tables, NOT streams.
#      Materialized Views do a full-refresh on every pipeline run — correct
#      behavior for slowly-changing dimensions.
# ===========================================================================

# ---------------------------------------------------------------------------
# dim_date — generated spine, no Silver source
# ---------------------------------------------------------------------------
@dp.materialized_view(
    name=f"{GOLD}.dim_date",
    comment=(
        "Date dimension — generated spine 2010-2050. "
        "Role-played x2 in fact_returns (return_date_key, order_date_key)."
    )
)
def dim_date():
    return spark.sql("""
        SELECT
            CAST(date_format(full_date, 'yyyyMMdd') AS INT)  AS date_key,
            full_date,
            date_format(full_date, 'EEEE')                  AS day_of_week,
            dayofmonth(full_date)                            AS day_num,
            weekofyear(full_date)                            AS week_num,
            month(full_date)                                 AS month_num,
            date_format(full_date, 'MMMM')                  AS month_name,
            quarter(full_date)                               AS quarter,
            year(full_date)                                  AS year,
            CASE WHEN dayofweek(full_date) IN (1, 7)
                 THEN TRUE ELSE FALSE END                    AS is_weekend
        FROM (
            SELECT explode(sequence(
                DATE '2010-01-01',
                DATE '2050-12-31',
                INTERVAL 1 DAY
            )) AS full_date
        )
    """)

# COMMAND ----------

# ---------------------------------------------------------------------------
# dim_customer
# ---------------------------------------------------------------------------
@dp.materialized_view(
    name=f"{GOLD}.dim_customer",
    comment="Customer dimension — 9 columns from silver_customers."
)
def dim_customer():
    return spark.sql(f"""
        SELECT DISTINCT
            customer_id,
            customer_email,
            customer_name,
            segment,
            country,
            city,
            state,
            postal_code,
            region
        FROM {SILVER}.silver_customers
        WHERE customer_id IS NOT NULL
    """)



# COMMAND ----------

# ---------------------------------------------------------------------------
# dim_vendor
# ---------------------------------------------------------------------------
@dp.materialized_view(
    name=f"{GOLD}.dim_vendor",
    comment="Vendor dimension — 7 vendors VEN01-VEN07 from silver_vendors."
)
def dim_vendor():
    return spark.sql(f"""
        SELECT DISTINCT
            vendor_id,
            vendor_name
        FROM {SILVER}.silver_vendors
        WHERE vendor_id IS NOT NULL
    """)


# COMMAND ----------

# ---------------------------------------------------------------------------
# dim_region — THIS WAS THE OFFENDING TABLE IN YOUR ERROR
# FIX: Using @dp.materialized_view() decorator
# ---------------------------------------------------------------------------
@dp.materialized_view(
    name=f"{GOLD}.dim_region",
    comment="Region dimension — distinct regions derived from silver_customers."
)
def dim_region():
    return spark.sql(f"""
        SELECT DISTINCT
            region    AS region_name,
            country
        FROM {SILVER}.silver_customers
        WHERE region IS NOT NULL
    """)


# COMMAND ----------

# ---------------------------------------------------------------------------
# dim_product — Gold-derived: category, sub_category
# ---------------------------------------------------------------------------
@dp.materialized_view(
    name=f"{GOLD}.dim_product",
    comment=(
        "Product dimension — from silver_products. "
        "Gold-derived: +category, +sub_category split from categories field."
    )
)
def dim_product():
    return spark.sql(f"""
        SELECT
            product_id,
            product_name,
            categories,
            brand,
            colors,
            sizes,
            upc,
            weight,
            dimension,
            manufacturer,
            TRIM(SPLIT(categories, '>')[0])                         AS category,
            CASE
                WHEN SIZE(SPLIT(categories, '>')) > 1
                THEN TRIM(SPLIT(categories, '>')[1])
                ELSE NULL
            END                                                     AS sub_category
        FROM {SILVER}.silver_products
        WHERE product_id IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY product_id) = 1
    """)


# COMMAND ----------

# ===========================================================================
# SECTION 2 — FACT TABLES
# FIX: Facts use create_streaming_table + append_flow with readStream
#      so new Silver rows are incrementally appended — no full reprocessing.
# ===========================================================================

# ---------------------------------------------------------------------------
# fact_sales
# FIX: silver_transactions read via spark.readStream (streaming source).
#      silver_orders_clean and silver_customers are small lookup tables —
#      read as batch inside the flow and broadcast-joined.
# ---------------------------------------------------------------------------
dp.create_streaming_table(
    name=f"{GOLD}.fact_sales",
    comment=(
        "Fact sales — grain: order_id + product_id. "
        "Incremental via readStream on silver_transactions."
    )
)

@dp.append_flow(target=f"{GOLD}.fact_sales")
def flow_fact_sales():
    # ── Streaming source (drives incrementalism) ──────────────────────────
    transactions = spark.readStream.table(f"{SILVER}.silver_transactions")

    # ── Batch lookups (broadcast-joined — small tables, full snapshot) ─────
    orders    = spark.read.table(f"{SILVER}.silver_orders_clean")
    customers = spark.read.table(f"{SILVER}.silver_customers").select("customer_id", "region")

    return (
        transactions.alias("t")
        .join(F.broadcast(orders.alias("o")),    on="order_id",    how="left")
        .join(F.broadcast(customers.alias("c")), on="customer_id", how="left")
        .filter(
            F.col("t.order_id").isNotNull() &
            F.col("t.product_id").isNotNull()
        )
        .select(
            # ── Foreign Keys ──────────────────────────────────────────────
            F.date_format(
                F.to_date(F.col("o.order_purchase_date")), "yyyyMMdd"
            ).cast("INT")                               .alias("date_key"),

            F.col("o.customer_id"),                     # FK → dim_customer
            F.col("t.product_id"),                      # FK → dim_product
            F.col("o.vendor_id"),                       # FK → dim_vendor
            F.col("c.region")                           .alias("region_name"),

            # ── Measures ──────────────────────────────────────────────────
            F.col("t.sales")                            .alias("sales_amount"),
            F.col("t.quantity"),
            F.col("t.discount"),
            F.col("t.profit"),
            F.col("t.payment_type"),
            F.col("t.payment_installments"),

            # ── Degenerate Dimensions ─────────────────────────────────────
            F.col("o.order_id"),
            F.col("o.ship_mode"),
            F.col("o.order_status"),
            F.col("o.order_approved_at"),
            F.col("o.order_delivered_carrier_date"),
            F.col("o.order_delivered_customer_date"),
            F.col("o.order_estimated_delivery_date")
        )
    )

# COMMAND ----------

# ---------------------------------------------------------------------------
# fact_returns
# FIX: silver_returns.return_date is already DATE type (not string)
#      No parsing needed - use directly for date_format and datediff
# ---------------------------------------------------------------------------
dp.create_streaming_table(
    name=f"{GOLD}.fact_returns",
    comment=(
        "Fact returns — grain: one return per order. "
        "dim_date role-played x2: return_date_key (role 1), order_date_key (role 2 ★). "
        "Gold-derived: is_approved, days_to_return."
    )
)

@dp.append_flow(target=f"{GOLD}.fact_returns")
def flow_fact_returns():
    # ── Streaming source ───────────────────────────────────────────────────
    returns   = spark.readStream.table(f"{SILVER}.silver_returns")

    # ── Batch lookups ──────────────────────────────────────────────────────
    orders    = spark.read.table(f"{SILVER}.silver_orders_clean") \
                    .select("order_id", "customer_id", "vendor_id", "order_purchase_date")
    customers = spark.read.table(f"{SILVER}.silver_customers") \
                    .select("customer_id", "region")

    return (
        returns.alias("r")
        .join(F.broadcast(orders.alias("o")),    on="order_id",    how="left")
        .join(F.broadcast(customers.alias("c")), on="customer_id", how="left")
        .filter(F.col("r.order_id").isNotNull())
        .select(
            # ── Foreign Keys ──────────────────────────────────────────────
            # return_date is already DATE type - no F.to_date() needed
            F.date_format(
                F.col("r.return_date"), "yyyyMMdd"
            ).cast("INT")                               .alias("return_date_key"),
                                                        # role 1 → dim_date

            F.date_format(
                F.to_date(F.col("o.order_purchase_date")), "yyyyMMdd"
            ).cast("INT")                               .alias("order_date_key"),
                                                        # role 2 ★ → dim_date

            F.col("o.customer_id"),                     # FK → dim_customer
            F.col("o.vendor_id"),                       # FK → dim_vendor
            F.col("c.region")                           .alias("region_name"),

            # ── Measures ──────────────────────────────────────────────────
            F.col("r.order_id"),
            F.col("r.return_reason"),
            # Cast DATE to string for yyyy-MM-dd output
            F.col("r.return_date").cast("string")       .alias("return_date"),
            F.col("r.refund_amount"),
            F.col("r.return_status"),

            # ── Gold-Derived Columns ───────────────────────────────────────
            F.when(F.lower(F.col("r.return_status")) == "approved", True)
             .otherwise(False)
             .cast("BOOLEAN")                           .alias("is_approved"),

            # return_date is DATE type - use directly in datediff
            F.datediff(
                F.col("r.return_date"),
                F.to_date(F.col("o.order_purchase_date"))
            )                                           .alias("days_to_return")
        )
    )

