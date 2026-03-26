# Databricks notebook source
# MAGIC %md
# MAGIC # UC2: Returns Fraud Investigator
# MAGIC
# MAGIC This notebook builds customer return-risk profiles from Gold tables, applies weighted anomaly rules, flags high-risk customers, generates investigation briefs with `databricks-gpt-oss-20b`, and writes `globalmart.gold.flagged_return_customers`.

# COMMAND ----------

from pyspark.sql import functions as F
 
CATALOG = "dbx_hck_glbl_mart"

SCHEMA = "gold"

TABLES = [

    f"{CATALOG}.{SCHEMA}.fact_returns",

    f"{CATALOG}.{SCHEMA}.dim_customer",

]

SAMPLE_ROWS = 5

NULL_PROFILE_LIMIT_COLS = 25  # keep output readable
 
def safe_table_info(full_name: str):

    print("\n" + "=" * 80)

    print(f"TABLE: {full_name}")

    try:

        df = spark.table(full_name)

        print("Row count:", df.count())

        df.printSchema()

        display(df.limit(SAMPLE_ROWS))
 
        # Null-profile for first N columns (helps pick correct rule fields)

        cols = df.columns[:NULL_PROFILE_LIMIT_COLS]

        if cols:

            nulls = df.select([

                F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols

            ])

            print(f"Null counts (first {len(cols)} cols):")

            display(nulls)

        else:

            print("No columns found.")

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
SOURCE_RETURNS = "globalmart.gold.fact_returns"
SOURCE_CUSTOMERS = "globalmart.gold.dim_customers"
TARGET_TABLE = "globalmart.gold.flagged_return_customers"

# Scoring setup
FLAG_THRESHOLD = 65
LOWER_THRESHOLD_DELTA = 10
HIGH_VALUE_RETURN_AMT = 200.0
BURST_WINDOW_DAYS = 30
SHORT_WINDOW_DAYS = 7

RULE_WEIGHTS = {
    "rule_no_matching_order": 30,
    "rule_high_return_frequency": 20,
    "rule_high_return_value": 20,
    "rule_multi_region_returns": 15,
    "rule_repeat_high_value_items": 15
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
print("dim_customers columns:", customers_df.columns)


# COMMAND ----------

def pick_col(df, candidates, default_expr=None):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return F.col(c)
    return default_expr

customer_key = pick_col(returns_df, ["customer_id", "cust_id", "customer_key"], F.lit(None))
return_id_col = pick_col(returns_df, ["return_id", "id"], F.monotonically_increasing_id())
return_value_col = pick_col(returns_df, ["return_amount", "return_value", "amount"], F.lit(0.0)).cast("double")
return_date_col = pick_col(returns_df, ["return_date", "created_at", "event_date"], F.current_date())
region_col = pick_col(returns_df, ["region", "return_region", "store_region"], F.lit("unknown"))
order_id_col = pick_col(returns_df, ["order_id", "original_order_id", "source_order_id"], F.lit(None))
matched_order_flag_col = pick_col(returns_df, ["has_matching_order", "matching_order_flag", "is_order_matched"], None)
item_id_col = pick_col(returns_df, ["product_id", "item_id", "sku"], F.lit("unknown_item"))

returns_norm = (
    returns_df
    .select(
        customer_key.cast("string").alias("customer_id"),
        return_id_col.cast("string").alias("return_id"),
        return_value_col.alias("return_value"),
        F.to_date(return_date_col).alias("return_date"),
        region_col.cast("string").alias("region"),
        order_id_col.cast("string").alias("order_id"),
        item_id_col.cast("string").alias("item_id"),
        (matched_order_flag_col.cast("boolean") if matched_order_flag_col is not None else F.lit(None).cast("boolean")).alias("has_matching_order")
    )
    .filter(F.col("customer_id").isNotNull())
)

# If explicit matching flag is unavailable, infer from null order_id.
returns_norm = returns_norm.withColumn(
    "is_no_matching_order",
    F.when(F.col("has_matching_order").isNotNull(), ~F.col("has_matching_order"))
     .otherwise(F.col("order_id").isNull())
)

returns_norm = returns_norm.withColumn("is_high_value_return", F.col("return_value") >= F.lit(HIGH_VALUE_RETURN_AMT))

latest_date = returns_norm.select(F.max("return_date").alias("mx")).first()["mx"]
if latest_date is None:
    latest_date = datetime.utcnow().date()

returns_window = returns_norm.withColumn(
    "is_recent_30d",
    F.col("return_date") >= F.date_sub(F.lit(latest_date), BURST_WINDOW_DAYS)
).withColumn(
    "is_recent_7d",
    F.col("return_date") >= F.date_sub(F.lit(latest_date), SHORT_WINDOW_DAYS)
)

repeat_high_value_items = (
    returns_window
    .filter(F.col("is_high_value_return"))
    .groupBy("customer_id", "item_id")
    .agg(F.count("return_id").alias("item_return_count"))
    .filter(F.col("item_return_count") >= 2)
    .groupBy("customer_id")
    .agg(F.count("item_id").alias("repeat_high_value_item_count"))
)

profiles = (
    returns_window
    .groupBy("customer_id")
    .agg(
        F.countDistinct("return_id").alias("total_returns"),
        F.round(F.sum(F.coalesce(F.col("return_value"), F.lit(0.0))), 2).alias("total_return_value"),
        F.round(F.avg(F.coalesce(F.col("return_value"), F.lit(0.0))), 2).alias("avg_return_value"),
        F.sum(F.when(F.col("is_high_value_return"), 1).otherwise(0)).alias("high_value_returns"),
        F.sum(F.when(F.col("is_no_matching_order"), 1).otherwise(0)).alias("no_matching_order_returns"),
        F.countDistinct("region").alias("distinct_regions"),
        F.sum(F.when(F.col("is_recent_30d"), 1).otherwise(0)).alias("returns_last_30d"),
        F.sum(F.when(F.col("is_recent_7d"), 1).otherwise(0)).alias("returns_last_7d")
    )
    .join(repeat_high_value_items, on="customer_id", how="left")
    .fillna({"repeat_high_value_item_count": 0})
    .withColumn("no_matching_order_ratio", F.when(F.col("total_returns") > 0, F.col("no_matching_order_returns") / F.col("total_returns")).otherwise(F.lit(0.0)))
    .withColumn("high_value_return_ratio", F.when(F.col("total_returns") > 0, F.col("high_value_returns") / F.col("total_returns")).otherwise(F.lit(0.0)))
)

customer_name_col = pick_col(customers_df, ["customer_name", "full_name", "name"], F.lit(None))
customer_region_col = pick_col(customers_df, ["region", "home_region", "customer_region"], F.lit(None))
customer_tier_col = pick_col(customers_df, ["customer_tier", "segment", "loyalty_tier"], F.lit(None))
customer_id_dim_col = pick_col(customers_df, ["customer_id", "cust_id", "customer_key"], F.lit(None))

customers_norm = customers_df.select(
    customer_id_dim_col.cast("string").alias("customer_id"),
    customer_name_col.cast("string").alias("customer_name"),
    customer_region_col.cast("string").alias("customer_region"),
    customer_tier_col.cast("string").alias("customer_tier")
).dropDuplicates(["customer_id"])

profiles = profiles.join(customers_norm, on="customer_id", how="left")
display(profiles.orderBy(F.desc("total_return_value")))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Weighted Anomaly Rules, Score, and Threshold Analysis

# COMMAND ----------

# Rule thresholds (defensible defaults; tune with data distribution)
THRESH_NO_MATCHING_ORDER_RATIO = 0.35
THRESH_TOTAL_RETURNS = 8
THRESH_TOTAL_RETURN_VALUE = 1000.0
THRESH_DISTINCT_REGIONS = 3
THRESH_REPEAT_HIGH_VALUE_ITEMS = 2

scored = (
    profiles
    .withColumn("rule_no_matching_order", (F.col("no_matching_order_ratio") >= F.lit(THRESH_NO_MATCHING_ORDER_RATIO)).cast("int"))
    .withColumn("rule_high_return_frequency", (F.col("total_returns") >= F.lit(THRESH_TOTAL_RETURNS)).cast("int"))
    .withColumn("rule_high_return_value", (F.col("total_return_value") >= F.lit(THRESH_TOTAL_RETURN_VALUE)).cast("int"))
    .withColumn("rule_multi_region_returns", (F.col("distinct_regions") >= F.lit(THRESH_DISTINCT_REGIONS)).cast("int"))
    .withColumn("rule_repeat_high_value_items", (F.col("repeat_high_value_item_count") >= F.lit(THRESH_REPEAT_HIGH_VALUE_ITEMS)).cast("int"))
    .withColumn("score_no_matching_order", F.col("rule_no_matching_order") * F.lit(RULE_WEIGHTS["rule_no_matching_order"]))
    .withColumn("score_high_return_frequency", F.col("rule_high_return_frequency") * F.lit(RULE_WEIGHTS["rule_high_return_frequency"]))
    .withColumn("score_high_return_value", F.col("rule_high_return_value") * F.lit(RULE_WEIGHTS["rule_high_return_value"]))
    .withColumn("score_multi_region_returns", F.col("rule_multi_region_returns") * F.lit(RULE_WEIGHTS["rule_multi_region_returns"]))
    .withColumn("score_repeat_high_value_items", F.col("rule_repeat_high_value_items") * F.lit(RULE_WEIGHTS["rule_repeat_high_value_items"]))
    .withColumn(
        "anomaly_score",
        F.col("score_no_matching_order")
        + F.col("score_high_return_frequency")
        + F.col("score_high_return_value")
        + F.col("score_multi_region_returns")
        + F.col("score_repeat_high_value_items")
    )
)

scored = scored.withColumn(
    "rules_violated",
    F.expr("filter(array("
           "IF(rule_no_matching_order = 1, 'rule_no_matching_order', NULL),"
           "IF(rule_high_return_frequency = 1, 'rule_high_return_frequency', NULL),"
           "IF(rule_high_return_value = 1, 'rule_high_return_value', NULL),"
           "IF(rule_multi_region_returns = 1, 'rule_multi_region_returns', NULL),"
           "IF(rule_repeat_high_value_items = 1, 'rule_repeat_high_value_items', NULL)"
           "), x -> x is not null)")
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
Write one concise investigation brief for a flagged customer using exactly 3 short sections:
1) Suspicious patterns
2) Possible innocent explanations
3) First verification actions

Customer context:
- customer_id: {customer_id}
- customer_name: {customer_name}
- customer_region: {customer_region}
- customer_tier: {customer_tier}
- anomaly_score: {anomaly_score}
- rules_violated: {rules_violated}
- total_returns: {total_returns}
- total_return_value: {total_return_value}
- avg_return_value: {avg_return_value}
- no_matching_order_returns: {no_matching_order_returns}
- no_matching_order_ratio: {no_matching_order_ratio}
- returns_last_30d: {returns_last_30d}
- returns_last_7d: {returns_last_7d}
- high_value_returns: {high_value_returns}
- high_value_return_ratio: {high_value_return_ratio}
- distinct_regions: {distinct_regions}
- repeat_high_value_item_count: {repeat_high_value_item_count}

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
        "no_matching_order_returns": int(r["no_matching_order_returns"] or 0),
        "no_matching_order_ratio": round(float(r["no_matching_order_ratio"] or 0.0), 4),
        "returns_last_30d": int(r["returns_last_30d"] or 0),
        "returns_last_7d": int(r["returns_last_7d"] or 0),
        "high_value_returns": int(r["high_value_returns"] or 0),
        "high_value_return_ratio": round(float(r["high_value_return_ratio"] or 0.0), 4),
        "distinct_regions": int(r["distinct_regions"] or 0),
        "repeat_high_value_item_count": int(r["repeat_high_value_item_count"] or 0)
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
    T.StructField("no_matching_order_returns", T.IntegerType(), True),
    T.StructField("no_matching_order_ratio", T.DoubleType(), True),
    T.StructField("returns_last_30d", T.IntegerType(), True),
    T.StructField("returns_last_7d", T.IntegerType(), True),
    T.StructField("high_value_returns", T.IntegerType(), True),
    T.StructField("high_value_return_ratio", T.DoubleType(), True),
    T.StructField("distinct_regions", T.IntegerType(), True),
    T.StructField("repeat_high_value_item_count", T.IntegerType(), True),
    T.StructField("anomaly_score", T.IntegerType(), True),
    T.StructField("rules_violated", T.StringType(), True),
    T.StructField("ai_investigation_brief", T.StringType(), True),
    T.StructField("generated_at", T.StringType(), True)
])

output_df = spark.createDataFrame(output_rows, schema=output_schema) if output_rows else spark.createDataFrame([], schema=output_schema)

spark.sql("CREATE SCHEMA IF NOT EXISTS globalmart.gold")
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

display(
    result_df
    .select("customer_id", "anomaly_score", "rules_violated", "ai_investigation_brief")
    .orderBy(F.desc("anomaly_score"))
    .limit(3)
)


# COMMAND ----------

# Deep-dive for highest-scoring customer
top_case = result_df.orderBy(F.desc("anomaly_score")).limit(1)
display(top_case)

print("Use this top case to verify brief alignment with violated rules and first-action recommendations.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Required Narrative Answers
# MAGIC
# MAGIC ### Rules, Weights, and Rationale
# MAGIC - `rule_no_matching_order` (weight 30) is strongest because returns lacking order linkage are difficult to validate and highly correlated with abuse or process manipulation.
# MAGIC - `rule_high_return_frequency` (20) captures unusually frequent return behavior.
# MAGIC - `rule_high_return_value` (20) captures elevated financial impact.
# MAGIC - `rule_multi_region_returns` (15) captures cross-region behavioral inconsistency.
# MAGIC - `rule_repeat_high_value_items` (15) captures repeated high-value return patterns.
# MAGIC
# MAGIC ### Threshold Choice and Workload Impact
# MAGIC - Default threshold is `65` to focus on stronger, multi-signal cases.
# MAGIC - Lowering threshold by 10 points (`55`) expands coverage but increases investigation load; the exact increase is printed in the threshold analysis cell.
# MAGIC
# MAGIC ### Flagged Case Volume Realism
# MAGIC - Compare flagged customer count to daily capacity (40-60 requests/day) and investigator staffing.
# MAGIC - Adjust threshold if queue size is operationally unrealistic.
# MAGIC
# MAGIC ### Highest-Risk Customer Brief Review
# MAGIC - Use top-case output above.
# MAGIC - Confirm brief references violated rules and includes concrete first verification actions.
