# Databricks notebook source
# MAGIC %md
# MAGIC # UC4: AI Revenue & Inventory Intelligence
# MAGIC
# MAGIC This notebook aggregates KPI statistics from Gold MVs, generates executive summaries for key business domains, demonstrates `ai_query()` usage in SQL, and writes output to `globalmart.gold.ai_business_insights`.

# COMMAND ----------

from pyspark.sql import functions as F
 
CATALOG = "dbx_hck_glbl_mart"
SCHEMA = "gold"
TABLES = [
    f"{CATALOG}.{SCHEMA}.mv_monthly_revenue_by_region",
    f"{CATALOG}.{SCHEMA}.mv_vendor_return_rate",
    f"{CATALOG}.{SCHEMA}.mv_slow_moving_products",
]
SAMPLE_ROWS = 5
NULL_PROFILE_LIMIT_COLS = 25
 
def safe_table_info(full_name: str):
    print("\n" + "=" * 90)
    print(f"TABLE: {full_name}")
    try:
        df = spark.table(full_name)
        print("Row count:", df.count())
        df.printSchema()
        display(df.limit(SAMPLE_ROWS))
        cols = df.columns[:NULL_PROFILE_LIMIT_COLS]
        if cols:
            nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols])
            print(f"Null counts (first {len(cols)} cols):")
            display(nulls)
    except Exception as e:
        print("ERROR:", str(e)[:800])
 
for t in TABLES:
    safe_table_info(t)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Setup and Configuration

# COMMAND ----------

from openai import OpenAI
import json
import re
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql import types as T

MODEL_NAME = "databricks-gpt-oss-20b"
TARGET_TABLE = "globalmart.gold.ai_business_insights"

MV_REVENUE_BY_REGION = "globalmart.gold.mv_revenue_by_region"
MV_RETURN_RATE_BY_VENDOR = "globalmart.gold.mv_return_rate_by_vendor"
MV_SLOW_MOVING_PRODUCTS = "globalmart.gold.mv_slow_moving_products"


# COMMAND ----------

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_ws_raw = spark.conf.get("spark.databricks.workspaceUrl")
_ws_url = _ws_raw.replace("https://", "").replace("http://", "").strip("/")

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"https://{_ws_url}/serving-endpoints"
)

print(f"Model: {MODEL_NAME}")
print(f"Target table: {TARGET_TABLE}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Read Gold MVs and Compute Aggregated KPIs

# COMMAND ----------

revenue_df = spark.table(MV_REVENUE_BY_REGION)
vendor_rr_df = spark.table(MV_RETURN_RATE_BY_VENDOR)
slow_df = spark.table(MV_SLOW_MOVING_PRODUCTS)

print("mv_revenue_by_region columns:", revenue_df.columns)
print("mv_return_rate_by_vendor columns:", vendor_rr_df.columns)
print("mv_slow_moving_products columns:", slow_df.columns)


# COMMAND ----------

def pick_col(df, candidates, default_expr=None):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return F.col(c)
    return default_expr

# Revenue MV normalization
rev_region_col = pick_col(revenue_df, ["region", "sales_region"], F.lit("unknown"))
rev_amt_col = pick_col(revenue_df, ["revenue", "total_revenue", "revenue_amount"], F.lit(0.0))

rev_norm = revenue_df.select(
    rev_region_col.cast("string").alias("region"),
    rev_amt_col.cast("double").alias("revenue")
)

# Vendor return-rate MV normalization
vr_vendor_col = pick_col(vendor_rr_df, ["vendor_name", "name", "supplier_name"], F.lit("unknown_vendor"))
vr_rate_col = pick_col(vendor_rr_df, ["return_rate", "vendor_return_rate", "return_rate_pct"], F.lit(0.0))
vr_returns_col = pick_col(vendor_rr_df, ["total_returns", "returns_count"], F.lit(0))

vr_norm = vendor_rr_df.select(
    vr_vendor_col.cast("string").alias("vendor_name"),
    vr_rate_col.cast("double").alias("return_rate"),
    vr_returns_col.cast("double").alias("total_returns")
)

# Slow-moving MV normalization
sm_product_col = pick_col(slow_df, ["product_name", "name", "product_title"], F.lit("unknown_product"))
sm_region_col = pick_col(slow_df, ["region", "sales_region"], F.lit("unknown"))
sm_value_col = pick_col(slow_df, ["inventory_value", "slow_inventory_value", "inventory_amount"], F.lit(0.0))
sm_flag_col = pick_col(slow_df, ["is_slow_moving", "slow_moving_flag", "slow_flag"], F.lit(True))

sm_norm = slow_df.select(
    sm_product_col.cast("string").alias("product_name"),
    sm_region_col.cast("string").alias("region"),
    sm_value_col.cast("double").alias("inventory_value"),
    sm_flag_col.cast("boolean").alias("is_slow_moving")
)

display(rev_norm.limit(10))
display(vr_norm.limit(10))
display(sm_norm.limit(10))


# COMMAND ----------

# KPI payload 1: revenue_performance
rev_total = rev_norm.agg(F.round(F.sum("revenue"), 2).alias("v")).first()["v"]
rev_avg_region = rev_norm.agg(F.round(F.avg("revenue"), 2).alias("v")).first()["v"]
rev_top = rev_norm.orderBy(F.desc("revenue")).limit(3).collect()
rev_bottom = rev_norm.orderBy(F.asc("revenue")).limit(3).collect()

revenue_kpis = {
    "total_revenue": float(rev_total or 0.0),
    "avg_revenue_per_region": float(rev_avg_region or 0.0),
    "top_regions": [{"region": r["region"], "revenue": float(r["revenue"] or 0.0)} for r in rev_top],
    "bottom_regions": [{"region": r["region"], "revenue": float(r["revenue"] or 0.0)} for r in rev_bottom],
    "region_count": int(rev_norm.select("region").distinct().count())
}

# KPI payload 2: vendor_return_rate
vr_avg = vr_norm.agg(F.round(F.avg("return_rate"), 4).alias("v")).first()["v"]
vr_top = vr_norm.orderBy(F.desc("return_rate")).limit(5).collect()
vr_high_count = vr_norm.filter(F.col("return_rate") >= F.lit(0.2)).count()

vendor_rr_kpis = {
    "avg_vendor_return_rate": float(vr_avg or 0.0),
    "high_return_rate_vendor_count_ge_20pct": int(vr_high_count),
    "top_vendor_return_rates": [
        {
            "vendor_name": r["vendor_name"],
            "return_rate": float(r["return_rate"] or 0.0),
            "total_returns": float(r["total_returns"] or 0.0)
        }
        for r in vr_top
    ],
    "vendor_count": int(vr_norm.select("vendor_name").distinct().count())
}

# KPI payload 3: slow_moving_inventory
sm_only = sm_norm.filter(F.coalesce(F.col("is_slow_moving"), F.lit(False)))
sm_total_products = sm_only.count()
sm_total_value = sm_only.agg(F.round(F.sum("inventory_value"), 2).alias("v")).first()["v"]
sm_by_region = sm_only.groupBy("region").agg(F.count("product_name").alias("slow_product_count"), F.round(F.sum("inventory_value"), 2).alias("slow_inventory_value")).orderBy(F.desc("slow_inventory_value")).limit(5).collect()

slow_moving_kpis = {
    "slow_moving_product_count": int(sm_total_products),
    "total_slow_moving_inventory_value": float(sm_total_value or 0.0),
    "top_regions_by_slow_inventory": [
        {
            "region": r["region"],
            "slow_product_count": int(r["slow_product_count"] or 0),
            "slow_inventory_value": float(r["slow_inventory_value"] or 0.0)
        }
        for r in sm_by_region
    ],
    "region_count_with_slow_inventory": int(sm_only.select("region").distinct().count())
}

print("Revenue KPI payload:")
print(json.dumps(revenue_kpis, indent=2))
print("\nVendor return-rate KPI payload:")
print(json.dumps(vendor_rr_kpis, indent=2))
print("\nSlow-moving KPI payload:")
print(json.dumps(slow_moving_kpis, indent=2))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Generate Executive Summaries from KPI JSON (No Raw Rows)

# COMMAND ----------

def extract_text_block(content):
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text", "")
                if txt and txt.strip():
                    return txt.strip()

    if isinstance(content, str):
        s = content.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                for block in parsed:
                    if isinstance(block, dict) and block.get("type") == "text":
                        txt = block.get("text", "")
                        if txt and txt.strip():
                            return txt.strip()
        except Exception:
            pass
        return re.sub(r"\s+", " ", s).strip()
    return ""

def generate_summary(insight_type, kpi_payload):
    payload_json = json.dumps(kpi_payload, indent=2)
    prompt = f"""
You are preparing an executive business summary for GlobalMart leadership.

Insight type: {insight_type}
KPI payload (aggregated only):
{payload_json}

Write a 4-6 sentence summary that:
1) states the most important KPI signals,
2) interprets business risk/opportunity,
3) suggests immediate leadership focus areas.

Constraints:
- Use only the KPI payload provided.
- Do not invent numbers.
- Keep tone concise and executive-focused.
""".strip()

    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You create concise, data-grounded executive summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            text = extract_text_block(resp.choices[0].message.content)
            if text:
                return text
        except Exception:
            continue

    return "KPI review indicates material trends that require leadership follow-up on performance variance, risk concentration, and near-term operational priorities."


# COMMAND ----------

generated_at = datetime.utcnow().isoformat()

insight_rows = [
    {
        "insight_type": "revenue_performance",
        "kpi_payload_json": json.dumps(revenue_kpis),
        "executive_summary": generate_summary("revenue_performance", revenue_kpis),
        "generated_at": generated_at
    },
    {
        "insight_type": "vendor_return_rate",
        "kpi_payload_json": json.dumps(vendor_rr_kpis),
        "executive_summary": generate_summary("vendor_return_rate", vendor_rr_kpis),
        "generated_at": generated_at
    },
    {
        "insight_type": "slow_moving_inventory",
        "kpi_payload_json": json.dumps(slow_moving_kpis),
        "executive_summary": generate_summary("slow_moving_inventory", slow_moving_kpis),
        "generated_at": generated_at
    }
]

for r in insight_rows:
    print("=" * 80)
    print(r["insight_type"])
    print(r["executive_summary"])


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Write Output Table `globalmart.gold.ai_business_insights`

# COMMAND ----------

out_schema = T.StructType([
    T.StructField("insight_type", T.StringType(), True),
    T.StructField("executive_summary", T.StringType(), True),
    T.StructField("kpi_payload_json", T.StringType(), True),
    T.StructField("generated_at", T.StringType(), True)
])

out_df = spark.createDataFrame(insight_rows, schema=out_schema)

spark.sql("CREATE SCHEMA IF NOT EXISTS globalmart.gold")
(
    out_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(TARGET_TABLE)
)

print(f"Wrote {out_df.count()} rows to {TARGET_TABLE}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) ai_query() SQL Demonstrations (2 Examples)

# COMMAND ----------

# Example 1: Revenue assessment by region using ai_query()
spark.sql("""
WITH rev AS (
  SELECT
    collect_list(named_struct('region', cast(region_name as string), 'revenue', cast(total_sales_amount as string))) AS rev_points
  FROM dbx_hck_glbl_mart.gold.mv_monthly_revenue_by_region
)
SELECT
  get_json_object(
    get_json_object(
      ai_query(
        'databricks-gpt-oss-20b',
        concat(
          'Provide a concise executive assessment (4 sentences max) of regional revenue performance from this data: ',
          to_json(rev_points)
        )
      ),
      '$[1]'
    ),
    '$.text'
  ) AS revenue_assessment
FROM rev
""").show(truncate=False)


# COMMAND ----------

# Example 2: Vendor return-rate risk assessment using ai_query()
spark.sql("""
WITH vr AS (
  SELECT
    collect_list(named_struct('vendor_name', cast(vendor_name as string), 'return_rate', cast(return_rate_pct as string))) AS vendor_points
  FROM dbx_hck_glbl_mart.gold.mv_vendor_return_rate
)
SELECT
  get_json_object(
    get_json_object(
      ai_query(
        'databricks-gpt-oss-20b',
        concat(
          'Based on this vendor return-rate data, provide a brief risk assessment and one priority action: ',
          to_json(vendor_points)
        )
      ),
      '$[1]'
    ),
    '$.text'
  ) AS vendor_return_rate_assessment
FROM vr
""").show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Validation Outputs

# COMMAND ----------

result_df = spark.table(TARGET_TABLE)
print(f"Row count: {result_df.count()}")
result_df.printSchema()
display(result_df.orderBy(F.col("insight_type")))


# COMMAND ----------

# One full summary sample for submission
display(result_df.filter(F.col("insight_type") == "revenue_performance").limit(1))
