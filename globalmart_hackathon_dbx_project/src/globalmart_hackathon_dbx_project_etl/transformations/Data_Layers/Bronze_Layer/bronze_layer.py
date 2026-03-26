# Databricks notebook source
# =============================================================================
# GLOBALMART — BRONZE LAYER INGESTION PIPELINE
# Spark Declarative Pipelines using Auto Loader (cloudFiles)
# =============================================================================
# 
# PURPOSE:
# Ingest 6 entities from cloud storage (Volumes) into Bronze Delta tables.
# Handles schema evolution, multiple regions, and tracks source lineage.
# 
# KEY DECISIONS:
# 1. Auto Loader (cloudFiles) - Tracks processed files, only loads new data
# 2. addNewColumns mode - Handles schema changes without breaking pipeline
# 3. Minimal schema hints - Allows empty directory startup, captures unknowns
# 4. recursiveFileLookup - Finds files in root + Region subdirectories
# 5. No transformations - Bronze layer preserves raw data exactly as received
# 
# TABLES CREATED:
# - bronze_customers, bronze_orders, bronze_transactions
# - bronze_returns, bronze_products, bronze_vendors
# =============================================================================

from pyspark import pipelines as dp
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CATALOG = "dbx_hck_glbl_mart"
SCHEMA = "bronze"
VOL = f"/Volumes/{CATALOG}/bronze/raw_landing"
# ---------------------------------------------------------------------------
# SCHEMA HINTS — Baseline schema with known column variants
# Auto Loader will add any new columns via schemaEvolutionMode="addNewColumns"
# ---------------------------------------------------------------------------
CUSTOMER_HINTS = (
    # ID variants: customer_id (R1,4,5), CustomerID (R2 - case-insensitive match), 
    # cust_id (R3), customer_identifier (R6)
    "customer_id STRING, CustomerID STRING, cust_id STRING, customer_identifier STRING, "
    # Email variants: customer_email (R1-4, missing in R5), email_address (R6)
    "customer_email STRING, email_address STRING, "
    # Name variants: customer_name (R1-5), full_name (R6)
    "customer_name STRING, full_name STRING, "
    # Segment variants: segment (R1-5), customer_segment (R6)
    "segment STRING, customer_segment STRING, "
    # Standard fields (note: R4 has city/state swapped - loaded as-is)
    "country STRING, city STRING, state STRING, postal_code STRING, region STRING"
)

ORDER_HINTS = (
    "order_id STRING, customer_id STRING, vendor_id STRING, "
    "ship_mode STRING, order_status STRING, "
    "order_purchase_date STRING, order_approved_at STRING, "
    "order_delivered_carrier_date STRING, order_delivered_customer_date STRING, "
    "order_estimated_delivery_date STRING"
)

TXN_HINTS = (
    # Case-insensitive: order_id matches Order_id (R2/R4) and Order_ID (ROOT)
    "Order_ID STRING, Product_ID STRING, "
    "Sales STRING, Quantity INTEGER, "
    # discount STRING - R4 sends "40%" as string, R2/ROOT send 0.2
    "discount STRING, " 
    # profit DOUBLE - R4 missing (NULL)
    "profit DOUBLE, "
    "payment_type STRING, payment_installments INTEGER"
)

# TXN_HINTS = (
#     "order_id STRING, product_id STRING, Sales DOUBLE, Quantity INTEGER, "
#     "discount STRING, profit DOUBLE, payment_type STRING, payment_installments INTEGER"
# )
RETURNS_HINTS = (
    # File 1 (R6) schema
    "order_id STRING, return_reason STRING, return_date STRING, "
    "refund_amount STRING, return_status STRING, "
    # File 2 (ROOT) schema - completely different column names
    "OrderId STRING, reason STRING, date_of_return STRING, "
    "amount DOUBLE, status STRING"
)

PRODUCT_HINTS = (
    "product_id STRING, product_name STRING, "
    "categories STRING, brand STRING, colors STRING, sizes STRING, "
    # upc must be STRING - contains scientific notation "6.40E+11"
    "upc STRING, "
    "weight STRING, dimension STRING, manufacturer STRING, "
    "dateAdded STRING, dateUpdated STRING, product_photos_qty INTEGER"
)

VENDOR_HINTS = "vendor_id STRING, vendor_name STRING"

# ---------------------------------------------------------------------------
# METADATA HELPER — Adds audit columns to every record
# ---------------------------------------------------------------------------
def add_metadata(df):
    """
    Adds 3 audit columns to track data lineage:
    - _source_file: Full file path (e.g., .../Region 1/customers_1.csv)
    - _load_timestamp: When the record was ingested
    - _region: Extracted region number (1-6) or "global" for root files
    """
    return (
        df.withColumn("_source_file", F.col("_metadata.file_path"))
          .withColumn("_load_timestamp", F.current_timestamp())
          .withColumn("_region",
            F.when(F.col("_metadata.file_path").rlike(r"(?i)region\s*\d+"),
                   F.regexp_extract(F.col("_metadata.file_path"), r"(?i)region\s*(\d+)", 1))
            .otherwise(F.lit("global")))
    )

# ---------------------------------------------------------------------------
# AUTO LOADER HELPER — Unified reader for all entities
# ---------------------------------------------------------------------------
def read_auto(path: str, fmt: str, schema_name: str, schema_hints: str = "", file_pattern: str = None):
    """
    Auto Loader configuration for incremental file ingestion.
    
    KEY FEATURES:
    - schemaLocation: Tracks processed files (no duplicates on re-run)
    - schemaEvolutionMode=addNewColumns: Handles new columns automatically
    - inferColumnTypes=false: Works with empty directories
    - recursiveFileLookup + pathGlobFilter: Finds files in root + subdirectories
    
    HANDLES Q6 (New "Birth Date" column):
    When a new column arrives, addNewColumns mode adds it to the schema.
    Previous records get NULL, new records get the value. Pipeline succeeds.
    
    HANDLES Q7 (Different columns per region):
    schemaHints provides baseline, addNewColumns captures variants.
    Each column variant becomes a separate column (except case-insensitive matches).
    """
    reader = (
        spark.readStream
             .format("cloudFiles")
             .option("cloudFiles.format", fmt)
             .option("cloudFiles.schemaLocation", f"{VOL}/_schemas/{schema_name}")
             .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
             .option("cloudFiles.inferColumnTypes", "false")
             .option("header", "true")
    )
    
    if schema_hints:
        reader = reader.option("cloudFiles.schemaHints", schema_hints)
    
    # Match files at any depth (root + Region subdirectories)
    if file_pattern:
        reader = reader.option("recursiveFileLookup", "true").option("pathGlobFilter", file_pattern)
    
    # JSON-specific: multiLine for array structures
    if fmt == "json":
        reader = reader.option("multiLine", "true")
        
    return reader.load(path).transform(add_metadata)

# ===========================================================================
# 6 ENTITIES — BRONZE LAYER STREAMING TABLES
# ===========================================================================

# 1. CUSTOMERS
# 6 files across regions with 4 ID variants, 2 email variants, 2 name variants
# R5 missing customer_email → NULL. R4 city/state swapped → loaded as-is
dp.create_streaming_table(
    name=f"{SCHEMA}.bronze_customers",
    comment="Raw customer data from all regions. Multiple column variants per region."
)
@dp.append_flow(target=f"{SCHEMA}.bronze_customers")
def flow_customers():
    return read_auto(VOL, "csv", "c_dyn", CUSTOMER_HINTS, "customers_*.csv")

# 2. ORDERS
# 3 files (Region 1, 3, 5). Dates as STRING with 2 formats (MM/DD/YYYY vs YYYY-MM-DD)
dp.create_streaming_table(
    name=f"{SCHEMA}.bronze_orders",
    comment="Raw order data. R5 missing order_estimated_delivery_date."
)
@dp.append_flow(target=f"{SCHEMA}.bronze_orders")
def flow_orders():
    return read_auto(VOL, "csv", "o_dyn", ORDER_HINTS, "orders_*.csv")

# 3. TRANSACTIONS
# 3 files (R2, R4, ROOT). R4: discount="40%" string, profit=NULL
dp.create_streaming_table(
    name=f"{SCHEMA}.bronze_transactions",
    comment="Raw transaction data. Order_id/Order_ID unified via case-insensitive matching."
)
@dp.append_flow(target=f"{SCHEMA}.bronze_transactions")
def flow_transactions():
    return read_auto(VOL, "csv", "t_dyn", TXN_HINTS, "transactions_*.csv")

# 4. RETURNS
# 2 files with COMPLETELY different column names (order_id vs OrderId, etc.)
# Both schemas included in hints to ensure proper capture
dp.create_streaming_table(
    name=f"{SCHEMA}.bronze_returns",
    comment="Raw returns data. Two files with different schemas - all columns captured."
)
@dp.append_flow(target=f"{SCHEMA}.bronze_returns")
def flow_returns():
    return read_auto(VOL, "json", "r_dyn", RETURNS_HINTS, "returns_*.json")

# 5. PRODUCTS
# 1 file (root level). upc as STRING to preserve scientific notation "6.40E+11"
dp.create_streaming_table(
    name=f"{SCHEMA}.bronze_products",
    comment="Global product catalog. UPC stored as STRING to prevent precision loss."
)
@dp.append_flow(target=f"{SCHEMA}.bronze_products")
def flow_products():
    return read_auto(VOL, "json", "p_dyn", PRODUCT_HINTS, "products*.json")

# 6. VENDORS
# 1 file (root level). Simple 2-column schema.
dp.create_streaming_table(
    name=f"{SCHEMA}.bronze_vendors",
    comment="Vendor master data. 7 vendors (VEN01-VEN07)."
)
@dp.append_flow(target=f"{SCHEMA}.bronze_vendors")
def flow_vendors():
    return read_auto(VOL, "csv", "v_dyn", VENDOR_HINTS, "vendors*.csv")
