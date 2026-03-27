# Databricks notebook source
# MAGIC %md
# MAGIC # UC2: Returns Fraud Investigator
# MAGIC
# MAGIC This notebook builds customer return-risk profiles from Gold tables, applies weighted anomaly rules, flags high-risk customers, generates investigation briefs with `databricks-gpt-oss-20b`, and writes `globalmart.gold.flagged_return_customers`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Setup and Configuration

# COMMAND ----------

# MAGIC %pip install openai

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from openai import OpenAI
import json
import re
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql import types as T

MODEL_NAME = "databricks-gpt-oss-20b"

# Workspace-specific catalogs/schemas
INPUT_CATALOG = "dbx_hck_glbl_mart"
INPUT_SCHEMA = "gold"
OUTPUT_CATALOG = "dbx_hck_glbl_mart"
OUTPUT_SCHEMA = "gold"

SOURCE_RETURNS = f"{INPUT_CATALOG}.{INPUT_SCHEMA}.fact_returns"
SOURCE_CUSTOMERS = f"{INPUT_CATALOG}.{INPUT_SCHEMA}.dim_customer"
TARGET_TABLE = f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.flagged_return_customers"

# Scoring setup
FLAG_THRESHOLD = 65
LOWER_THRESHOLD_DELTA = 10
BURST_WINDOW_DAYS = 30
SHORT_WINDOW_DAYS = 7

RULE_WEIGHTS = {
    "rule_high_return_frequency": 20,
    "rule_high_return_value": 25,
    "rule_negative_days_to_return": 20,
    "rule_multi_region_returns": 15,
    "rule_high_non_approved_ratio": 20
}


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
# MAGIC ## 2) Source Loading and Profile Feature Engineering

# COMMAND ----------

returns_df = spark.table(SOURCE_RETURNS)
customers_df = spark.table(SOURCE_CUSTOMERS)

print("fact_returns columns:", returns_df.columns)
print("dim_customer columns:", customers_df.columns)


# COMMAND ----------

def pick_col(df, candidates, default_expr=None):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return F.col(c)
    return default_expr

# fact_returns (dbx_hck_glbl_mart.gold.fact_returns) columns we rely on:
# customer_id, order_id, region_name, return_reason, return_date (string), refund_amount, return_status, is_approved, days_to_return

returns_norm = (
    returns_df
    .select(
        F.col("customer_id").cast("string").alias("customer_id"),
        F.col("order_id").cast("string").alias("order_id"),
        F.col("vendor_id").cast("string").alias("vendor_id"),
        F.col("region_name").cast("string").alias("region_name"),
        F.col("return_reason").cast("string").alias("return_reason"),
        F.to_date(F.col("return_date").cast("string")).alias("return_date"),
        F.col("refund_amount").cast("double").alias("return_value"),
        F.col("return_status").cast("string").alias("return_status"),
        F.col("is_approved").cast("boolean").alias("is_approved"),
        F.col("days_to_return").cast("int").alias("days_to_return"),
    )
    .filter(F.col("customer_id").isNotNull())
)

# Establish time anchor for burst windows
latest_date = returns_norm.select(F.max("return_date").alias("mx")).first()["mx"]
if latest_date is None:
    latest_date = datetime.utcnow().date()

returns_window = (
    returns_norm
    .withColumn("is_recent_30d", F.col("return_date") >= F.date_sub(F.lit(latest_date), BURST_WINDOW_DAYS))
    .withColumn("is_recent_7d", F.col("return_date") >= F.date_sub(F.lit(latest_date), SHORT_WINDOW_DAYS))
    .withColumn("is_negative_days_to_return", (F.col("days_to_return") < F.lit(0)).cast("int"))
    .withColumn(
        "is_non_approved",
        (
            (~F.col("is_approved"))
            | (F.lower(F.col("return_status")).isin("rejected", "pending"))
        ).cast("int"),
    )
)

profiles = (
    returns_window
    .groupBy("customer_id")
    .agg(
        F.count(F.lit(1)).alias("total_returns"),
        F.round(F.sum(F.coalesce(F.col("return_value"), F.lit(0.0))), 2).alias("total_return_value"),
        F.round(F.avg(F.coalesce(F.col("return_value"), F.lit(0.0))), 2).alias("avg_return_value"),
        F.countDistinct("region_name").alias("distinct_regions"),
        F.sum(F.col("is_negative_days_to_return")).alias("negative_days_to_return_count"),
        F.sum(F.col("is_non_approved")).alias("non_approved_returns"),
        F.sum(F.when(F.col("is_recent_30d"), 1).otherwise(0)).alias("returns_last_30d"),
        F.sum(F.when(F.col("is_recent_7d"), 1).otherwise(0)).alias("returns_last_7d"),
    )
    .withColumn(
        "negative_days_to_return_ratio",
        F.when(F.col("total_returns") > 0, F.col("negative_days_to_return_count") / F.col("total_returns")).otherwise(F.lit(0.0)),
    )
    .withColumn(
        "non_approved_ratio",
        F.when(F.col("total_returns") > 0, F.col("non_approved_returns") / F.col("total_returns")).otherwise(F.lit(0.0)),
    )
)

# dim_customer columns: customer_id, customer_email, customer_name, segment, country, city, state, postal_code, region
customer_id_dim_col = pick_col(customers_df, ["customer_id"], F.lit(None))
customer_name_col = pick_col(customers_df, ["customer_name"], F.lit(None))
customer_region_col = pick_col(customers_df, ["region"], F.lit(None))
customer_segment_col = pick_col(customers_df, ["segment"], F.lit(None))

customers_norm = (
    customers_df
    .select(
        customer_id_dim_col.cast("string").alias("customer_id"),
        customer_name_col.cast("string").alias("customer_name"),
        customer_region_col.cast("string").alias("customer_region"),
        customer_segment_col.cast("string").alias("customer_tier"),
    )
    .dropDuplicates(["customer_id"])
)

profiles = profiles.join(customers_norm, on="customer_id", how="left")
display(profiles.orderBy(F.desc("total_return_value")))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Weighted Anomaly Rules, Score, and Threshold Analysis

# COMMAND ----------

# Rule thresholds (defensible defaults; tune with data distribution)
THRESH_TOTAL_RETURNS = 8
THRESH_TOTAL_RETURN_VALUE = 1000.0
THRESH_NEGATIVE_DAYS_RATIO = 0.15
THRESH_DISTINCT_REGIONS = 3
THRESH_NON_APPROVED_RATIO = 0.50

scored = (
    profiles
    .withColumn("rule_high_return_frequency", (F.col("total_returns") >= F.lit(THRESH_TOTAL_RETURNS)).cast("int"))
    .withColumn("rule_high_return_value", (F.col("total_return_value") >= F.lit(THRESH_TOTAL_RETURN_VALUE)).cast("int"))
    .withColumn("rule_negative_days_to_return", (F.col("negative_days_to_return_ratio") >= F.lit(THRESH_NEGATIVE_DAYS_RATIO)).cast("int"))
    .withColumn("rule_multi_region_returns", (F.col("distinct_regions") >= F.lit(THRESH_DISTINCT_REGIONS)).cast("int"))
    .withColumn("rule_high_non_approved_ratio", (F.col("non_approved_ratio") >= F.lit(THRESH_NON_APPROVED_RATIO)).cast("int"))
    .withColumn("score_high_return_frequency", F.col("rule_high_return_frequency") * F.lit(RULE_WEIGHTS["rule_high_return_frequency"]))
    .withColumn("score_high_return_value", F.col("rule_high_return_value") * F.lit(RULE_WEIGHTS["rule_high_return_value"]))
    .withColumn("score_negative_days_to_return", F.col("rule_negative_days_to_return") * F.lit(RULE_WEIGHTS["rule_negative_days_to_return"]))
    .withColumn("score_multi_region_returns", F.col("rule_multi_region_returns") * F.lit(RULE_WEIGHTS["rule_multi_region_returns"]))
    .withColumn("score_high_non_approved_ratio", F.col("rule_high_non_approved_ratio") * F.lit(RULE_WEIGHTS["rule_high_non_approved_ratio"]))
    .withColumn(
        "anomaly_score",
        F.col("score_high_return_frequency")
        + F.col("score_high_return_value")
        + F.col("score_negative_days_to_return")
        + F.col("score_multi_region_returns")
        + F.col("score_high_non_approved_ratio")
    )
)

scored = scored.withColumn(
    "rules_violated",
    F.expr(
        "filter(array("
        "IF(rule_high_return_frequency = 1, 'rule_high_return_frequency', NULL),"
        "IF(rule_high_return_value = 1, 'rule_high_return_value', NULL),"
        "IF(rule_negative_days_to_return = 1, 'rule_negative_days_to_return', NULL),"
        "IF(rule_multi_region_returns = 1, 'rule_multi_region_returns', NULL),"
        "IF(rule_high_non_approved_ratio = 1, 'rule_high_non_approved_ratio', NULL)"
        "), x -> x is not null)"
    )
)

scored = scored.withColumn("is_flagged", F.col("anomaly_score") >= F.lit(FLAG_THRESHOLD))

display(scored.orderBy(F.desc("anomaly_score")))


# COMMAND ----------

total_customers = scored.select("customer_id").distinct().count()
flagged_count = scored.filter(F.col("is_flagged")).count()
flagged_pct = (flagged_count / total_customers * 100.0) if total_customers else 0.0

lower_threshold = max(0, FLAG_THRESHOLD - LOWER_THRESHOLD_DELTA)
flagged_count_lower = scored.filter(F.col("anomaly_score") >= F.lit(lower_threshold)).count()
added_cases = flagged_count_lower - flagged_count

print(f"Total customers scored: {total_customers}")
print(f"Flagged @ threshold {FLAG_THRESHOLD}: {flagged_count} ({flagged_pct:.2f}%)")
print(f"Flagged @ threshold {lower_threshold}: {flagged_count_lower} (additional cases: {added_cases})")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) AI Investigation Briefs for Flagged Customers Only

# COMMAND ----------

PROMPT_TEMPLATE = """
You are an investigation assistant for GlobalMart returns operations.

Generate a strictly formatted investigation brief using EXACTLY the structure below.

OUTPUT FORMAT (follow exactly, do not deviate):

**Investigation Brief: {customer_id} ({customer_name})**

1. Suspicious patterns

* Bullet 1 (must include numeric values from context)
* Bullet 2
* Bullet 3

2. Possible innocent explanations

* Bullet 1
* Bullet 2
* Bullet 3

3. First verification actions

* Action 1 (must be actionable and specific)
* Action 2
* Action 3

STRICT RULES:

* The first line MUST be exactly: **Investigation Brief - {customer_id} ({customer_name})**
* Do NOT add any extra text before or after the header.
* Use ONLY bullet points (no paragraphs, no numbering inside sections).
* Each section must have EXACTLY 3 bullets (no more, no less).
* Each bullet must be ONE sentence only.
* Always include numeric values where available (e.g., %, counts, $).
* Do NOT repeat the same metric across multiple bullets unless necessary.
* Do NOT add introductions, conclusions, or extra headings.
* Do NOT mention "customer context" explicitly.
* Keep tone operational, concise, and consistent.
* Do NOT make legal conclusions.

Customer context:

* customer_id: {customer_id}
* customer_name: {customer_name}
* customer_region: {customer_region}
* customer_tier: {customer_tier}
* anomaly_score: {anomaly_score}
* rules_violated: {rules_violated}
* total_returns: {total_returns}
* total_return_value: {total_return_value}
* avg_return_value: {avg_return_value}
* returns_last_30d: {returns_last_30d}
* returns_last_7d: {returns_last_7d}
* distinct_regions: {distinct_regions}
* negative_days_to_return_count: {negative_days_to_return_count}
* negative_days_to_return_ratio: {negative_days_to_return_ratio}
* non_approved_returns: {non_approved_returns}
* non_approved_ratio: {non_approved_ratio}

Requirements:
- Use concrete values from the provided context.
- Do not make legal conclusions.
- Keep the final brief operational and actionable for returns managers.
""".strip()

def extract_text_block(content):
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text and text.strip():
                    return text.strip()

    if isinstance(content, str):
        raw = content.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for block in parsed:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text and text.strip():
                            return text.strip()
        except Exception:
            pass
        return re.sub(r"\s+", " ", raw).strip()

    return ""

def generate_brief(row_dict):
    prompt = PROMPT_TEMPLATE.format(**row_dict)
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You produce concise, evidence-based fraud investigation briefs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            content = resp.choices[0].message.content
            text = extract_text_block(content)
            if text:
                return text
        except Exception:
            continue

    return (
        "Suspicious patterns: This customer triggered multiple fraud-indicator rules based on return behavior and profile metrics. "
        "Possible innocent explanations: Operational errors, delayed order synchronization, or legitimate dissatisfaction could explain some anomalies. "
        "First verification actions: Verify order matching records, inspect recent return timelines, and review high-value item return evidence before escalation."
    )


# COMMAND ----------

flagged_df = scored.filter(F.col("is_flagged"))
flagged_rows = flagged_df.collect()
generated_at = datetime.utcnow().isoformat()

output_rows = []
for r in flagged_rows:
    payload = {
        "customer_id": r["customer_id"],
        "customer_name": r["customer_name"],
        "customer_region": r["customer_region"],
        "customer_tier": r["customer_tier"],
        "anomaly_score": int(r["anomaly_score"]),
        "rules_violated": ", ".join(r["rules_violated"] or []),
        "total_returns": int(r["total_returns"] or 0),
        "total_return_value": float(r["total_return_value"] or 0.0),
        "avg_return_value": float(r["avg_return_value"] or 0.0),
        "returns_last_30d": int(r["returns_last_30d"] or 0),
        "returns_last_7d": int(r["returns_last_7d"] or 0),
        "distinct_regions": int(r["distinct_regions"] or 0),
        "negative_days_to_return_count": int(r["negative_days_to_return_count"] or 0),
        "negative_days_to_return_ratio": round(float(r["negative_days_to_return_ratio"] or 0.0), 4),
        "non_approved_returns": int(r["non_approved_returns"] or 0),
        "non_approved_ratio": round(float(r["non_approved_ratio"] or 0.0), 4),
    }
    brief = generate_brief(payload)

    output_rows.append({
        **payload,
        "ai_investigation_brief": brief,
        "generated_at": generated_at
    })

output_schema = T.StructType([
    T.StructField("customer_id", T.StringType(), True),
    T.StructField("customer_name", T.StringType(), True),
    T.StructField("customer_region", T.StringType(), True),
    T.StructField("customer_tier", T.StringType(), True),
    T.StructField("total_returns", T.IntegerType(), True),
    T.StructField("total_return_value", T.DoubleType(), True),
    T.StructField("avg_return_value", T.DoubleType(), True),
    T.StructField("returns_last_30d", T.IntegerType(), True),
    T.StructField("returns_last_7d", T.IntegerType(), True),
    T.StructField("distinct_regions", T.IntegerType(), True),
    T.StructField("negative_days_to_return_count", T.IntegerType(), True),
    T.StructField("negative_days_to_return_ratio", T.DoubleType(), True),
    T.StructField("non_approved_returns", T.IntegerType(), True),
    T.StructField("non_approved_ratio", T.DoubleType(), True),
    T.StructField("anomaly_score", T.IntegerType(), True),
    T.StructField("rules_violated", T.StringType(), True),
    T.StructField("ai_investigation_brief", T.StringType(), True),
    T.StructField("generated_at", T.StringType(), True)
])

output_df = spark.createDataFrame(output_rows, schema=output_schema) if output_rows else spark.createDataFrame([], schema=output_schema)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}")
(
    output_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(TARGET_TABLE)
)

print(f"Wrote {output_df.count()} rows to {TARGET_TABLE}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Validation and Submission Evidence Cells

# COMMAND ----------

result_df = spark.table(TARGET_TABLE)
print(f"Row count: {result_df.count()}")
result_df.printSchema()

display(result_df.orderBy(F.desc("anomaly_score")))

# Evidence: non-truncated briefs
_topN = 3
rows = (
    result_df
    .select("customer_id", "anomaly_score", "rules_violated", "ai_investigation_brief")
    .orderBy(F.desc("anomaly_score"))
    .limit(_topN)
    .collect()
)

for i, r in enumerate(rows, start=1):
    print("=" * 80)
    print(f"Top case #{i}")
    print(f"customer_id: {r['customer_id']}")
    print(f"anomaly_score: {r['anomaly_score']}")
    print(f"rules_violated: {r['rules_violated']}")
    print("ai_investigation_brief:")
    print(r["ai_investigation_brief"])


# COMMAND ----------

# Deep-dive for highest-scoring customer
top_case = result_df.orderBy(F.desc("anomaly_score")).limit(1)
display(top_case)

print("Use this top case to verify brief alignment with violated rules and first-action recommendations.")
