# Databricks notebook source
# MAGIC %md
# MAGIC # UC1: AI Data Quality Reporter
# MAGIC
# MAGIC This notebook reads Silver quarantine records, groups rejections by issue type, generates finance-friendly explanations using `databricks-gpt-oss-20b`, and writes the final report to `globalmart.gold.dq_audit_report`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Setup and Configuration

# COMMAND ----------

# MAGIC %pip install openai

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
INPUT_SCHEMA = "silver"
OUTPUT_CATALOG = "dbx_hck_glbl_mart"
OUTPUT_SCHEMA = "gold"

TARGET_TABLE = f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.dq_audit_report"

# Include both `quarantine` and the dataset's misspelling `quaruntine`
CANDIDATE_QUARANTINE_TABLES = [
    f"{INPUT_CATALOG}.{INPUT_SCHEMA}.silver_customers_quaruntine",
    f"{INPUT_CATALOG}.{INPUT_SCHEMA}.silver_orders_quaruntine",
    f"{INPUT_CATALOG}.{INPUT_SCHEMA}.silver_transactions_quarantine",
    f"{INPUT_CATALOG}.{INPUT_SCHEMA}.silver_products_quarantine",
    f"{INPUT_CATALOG}.{INPUT_SCHEMA}.silver_returns_quarantine",
    f"{INPUT_CATALOG}.{INPUT_SCHEMA}.silver_vendors_quarantine",
]

MAX_SAMPLES_PER_GROUP = 5

# COMMAND ----------

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_ws_raw = spark.conf.get("spark.databricks.workspaceUrl")
_ws_url = _ws_raw.replace("https://", "").replace("http://", "").strip("/")

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"https://{_ws_url}/serving-endpoints"
)

print(f"Using model: {MODEL_NAME}")
print(f"Output table: {TARGET_TABLE}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Discover and Load Quarantine Data

# COMMAND ----------

def table_exists(full_name: str) -> bool:
    try:
        spark.table(full_name).limit(1).count()
        return True
    except Exception:
        return False

def discover_quarantine_tables(catalog: str, schema: str):
    discovered = []
    try:
        rows = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
        for r in rows:
            t = r["tableName"]
            tl = t.lower()
            if ("quarantine" in tl) or ("quaruntine" in tl):
                discovered.append(f"{catalog}.{schema}.{t}")
    except Exception:
        pass
    return discovered

selected_tables = [t for t in CANDIDATE_QUARANTINE_TABLES if table_exists(t)]
for t in discover_quarantine_tables(INPUT_CATALOG, INPUT_SCHEMA):
    if t not in selected_tables and table_exists(t):
        selected_tables.append(t)

if not selected_tables:
    raise ValueError(
        f"No quarantine tables found in {INPUT_CATALOG}.{INPUT_SCHEMA}. "
        "Update CANDIDATE_QUARANTINE_TABLES for your workspace."
    )

print("Quarantine tables used:")
for t in selected_tables:
    print(f" - {t}")


# COMMAND ----------

def first_existing_col(df, candidates, default_literal=None):
    existing = set(df.columns)
    for c in candidates:
        if c in existing:
            return F.col(c)
    if default_literal is not None:
        return F.lit(default_literal)
    return None

def infer_entity_name(table_name: str) -> str:
    n = table_name.lower()
    if "customer" in n:
        return "customers"
    if "order" in n:
        return "orders"
    if "product" in n:
        return "products"
    if "return" in n:
        return "returns"
    if "vendor" in n:
        return "vendors"
    if "transaction" in n or "payment" in n:
        return "transactions"
    return "unknown"

# Map rule names (error_reason) to the most likely affected field.
# Extend this dict as you discover more error_reason values.
ERROR_REASON_TO_FIELD = {
    "valid_ship_mode": "ship_mode",
    "valid_upc": "upc",
    "valid_return_reason": "return_reason",
    "valid_refund_amount": "refund_amount",
}

normalized_frames = []
for t in selected_tables:
    raw = spark.table(t)
    entity_name = infer_entity_name(t)

    # In your quarantine tables, `error_reason` is the rule name (best proxy for issue_type)
    issue_col = first_existing_col(raw, ["issue_type", "rule_name", "dq_rule", "error_type", "error_code", "error_reason"], "unknown_issue")

    # Choose an entity-appropriate record id
    record_id_col = first_existing_col(
        raw,
        [
            "record_id",
            "id",
            "order_id",
            "customer_id",
            "product_id",
            "vendor_id",
            "transaction_id",
            "upc",
        ],
        None,
    )

    if record_id_col is None:
        record_id_col = F.sha2(
            F.concat_ws(
                "||",
                *[F.coalesce(F.col(c).cast("string"), F.lit("")) for c in raw.columns[: min(len(raw.columns), 8)]],
            ),
            256,
        )

    # Derive field_name from issue_type using the mapping when possible.
    issue_lc = F.lower(issue_col.cast("string"))
    field_name_expr = F.coalesce(
        *[
            F.when(issue_lc == F.lit(k), F.lit(v))
            for k, v in ERROR_REASON_TO_FIELD.items()
        ],
        first_existing_col(raw, ["field_name", "column_name", "failed_field", "field"], "unknown_field").cast("string"),
    )

    # Choose trigger_value based on derived field_name; fall back to common rejected-value columns.
    trigger_candidates = [
        "rejected_value",
        "invalid_value",
        "failed_value",
        "value",
    ]
    generic_trigger_col = first_existing_col(raw, trigger_candidates, None)

    # Build a small set of likely “offending value” columns (safe even if missing)
    ship_mode_col = first_existing_col(raw, ["ship_mode"], None)
    upc_col = first_existing_col(raw, ["upc", "upc_cleaned"], None)
    refund_amount_col = first_existing_col(raw, ["refund_amount"], None)
    return_reason_col = first_existing_col(raw, ["return_reason"], None)

    trigger_value_expr = F.coalesce(
        F.when(field_name_expr == F.lit("ship_mode"), ship_mode_col.cast("string") if ship_mode_col is not None else F.lit(None).cast("string")),
        F.when(field_name_expr == F.lit("upc"), upc_col.cast("string") if upc_col is not None else F.lit(None).cast("string")),
        F.when(field_name_expr == F.lit("refund_amount"), refund_amount_col.cast("string") if refund_amount_col is not None else F.lit(None).cast("string")),
        F.when(field_name_expr == F.lit("return_reason"), return_reason_col.cast("string") if return_reason_col is not None else F.lit(None).cast("string")),
        generic_trigger_col.cast("string") if generic_trigger_col is not None else F.lit(None).cast("string"),
    )

    nf = (
        raw.select(
            F.lit(entity_name).alias("entity_name"),
            field_name_expr.cast("string").alias("field_name"),
            issue_col.cast("string").alias("issue_type"),
            record_id_col.cast("string").alias("record_id"),
            trigger_value_expr.alias("trigger_value"),
        )
        .withColumn("source_table", F.lit(t))
    )

    normalized_frames.append(nf)

dq_rejects = normalized_frames[0]
for nf in normalized_frames[1:]:
    dq_rejects = dq_rejects.unionByName(nf, allowMissingColumns=True)

dq_rejects = dq_rejects.filter(F.col("issue_type").isNotNull())
display(dq_rejects.limit(20))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Group Rejections by Issue Type (No Per-Row Generation)

# COMMAND ----------

grouped_df = (
    dq_rejects
    .groupBy("entity_name", "field_name", "issue_type")
    .agg(
        F.countDistinct("record_id").alias("rejected_count"),
        F.slice(F.array_distinct(F.collect_list(F.col("trigger_value"))), 1, MAX_SAMPLES_PER_GROUP).alias("sample_trigger_values"),
        F.array_sort(F.array_distinct(F.collect_list(F.col("source_table")))).alias("source_tables")
    )
    .orderBy(F.desc("rejected_count"))
)

display(grouped_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Prompt, LLM Call, and Structured Response Parsing

# COMMAND ----------

PROMPT_TEMPLATE = """
You are assisting a finance audit team at GlobalMart.
Write exactly one plain-English paragraph (3 to 4 sentences) for this grouped data-quality issue.

Context:
- Entity: {entity_name}
- Affected field: {field_name}
- Issue type (validation rule name): {issue_type}
- Rejected record count: {rejected_count}
- Sample triggering values/patterns: {sample_trigger_values}
- Source quarantine tables: {source_tables}

Your paragraph must include:
1) What the issue is and what triggered rejection (use examples from sample triggering values).
2) Why records cannot be accepted into analytics.
3) What finance report, audit figure, or business decision is at risk.

Constraints:
- Avoid technical code terms and stack traces.
- Be specific to the given field, issue type, and count.
- Keep a professional tone for finance auditors.
""".strip()

def extract_text_block(content):
    # databricks-gpt-oss-20b may return JSON-like list: [{"type":"reasoning"...}, {"type":"text","text":"..."}]
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                txt = b.get("text", "")
                if txt and txt.strip():
                    return txt.strip()

    if isinstance(content, str):
        s = content.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                for b in parsed:
                    if isinstance(b, dict) and b.get("type") == "text":
                        txt = b.get("text", "")
                        if txt and txt.strip():
                            return txt.strip()
        except Exception:
            pass

        # Fallback: strip obvious JSON wrappers if present and return raw text.
        return re.sub(r"\s+", " ", s).strip()

    return ""

def generate_explanation(entity_name, field_name, issue_type, rejected_count, sample_trigger_values, source_tables):
    prompt = PROMPT_TEMPLATE.format(
        entity_name=entity_name,
        field_name=field_name,
        issue_type=issue_type,
        rejected_count=rejected_count,
        sample_trigger_values=sample_trigger_values,
        source_tables=source_tables
    )

    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You produce clear audit-focused business explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            msg_content = resp.choices[0].message.content
            text = extract_text_block(msg_content)
            if text:
                return text
        except Exception:
            continue

    return (
        f"The {field_name} field for {entity_name} has {rejected_count} rejected records under issue type {issue_type}. "
        "These records are excluded from analytics because they fail data quality validation rules. "
        "This can distort audit totals, trend reporting, and finance decisions that rely on complete and valid records."
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Build Report Rows and Write Gold Output

# COMMAND ----------

group_rows = grouped_df.collect()
report_rows = []
generated_ts = datetime.utcnow().isoformat()

for r in group_rows:
    entity_name = (r["entity_name"] or "unknown").strip()
    field_name = (r["field_name"] or "unknown_field").strip()
    issue_type = (r["issue_type"] or "unknown_issue").strip()
    rejected_count = int(r["rejected_count"] or 0)
    sample_values = [v for v in (r["sample_trigger_values"] or []) if v is not None][:MAX_SAMPLES_PER_GROUP]
    source_tables = [v for v in (r["source_tables"] or []) if v is not None]

    explanation = generate_explanation(
        entity_name=entity_name,
        field_name=field_name,
        issue_type=issue_type,
        rejected_count=rejected_count,
        sample_trigger_values=sample_values,
        source_tables=source_tables
    )

    report_rows.append({
        "entity_name": entity_name,
        "field_name": field_name,
        "issue_type": issue_type,
        "rejected_count": rejected_count,
        "sample_trigger_values": sample_values,
        "source_tables": source_tables,
        "ai_business_impact_explanation": explanation,
        "generated_at": generated_ts
    })

schema = T.StructType([
    T.StructField("entity_name", T.StringType(), True),
    T.StructField("field_name", T.StringType(), True),
    T.StructField("issue_type", T.StringType(), True),
    T.StructField("rejected_count", T.LongType(), True),
    T.StructField("sample_trigger_values", T.ArrayType(T.StringType()), True),
    T.StructField("source_tables", T.ArrayType(T.StringType()), True),
    T.StructField("ai_business_impact_explanation", T.StringType(), True),
    T.StructField("generated_at", T.StringType(), True)
])

report_df = spark.createDataFrame(report_rows, schema=schema)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}")
(
    report_df
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(TARGET_TABLE)
)

print(f"Wrote {report_df.count()} rows to {TARGET_TABLE}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Validation and Sample Output

# COMMAND ----------

gold_df = spark.table(TARGET_TABLE)
print(f"Row count: {gold_df.count()}")
gold_df.printSchema()

display(
    gold_df.select(
        "entity_name",
        "field_name",
        "issue_type",
        "rejected_count",
        "ai_business_impact_explanation",
        "generated_at"
    ).orderBy(F.desc("rejected_count"))
)

# Show at least 2 full explanations
display(
    gold_df.select("entity_name", "field_name", "issue_type", "rejected_count", "ai_business_impact_explanation")
    .orderBy(F.desc("rejected_count"))
    .limit(2)
)

quality_df = (
    gold_df
    .withColumn("has_risk_language", F.lower(F.col("ai_business_impact_explanation")).rlike("risk|impact|audit|report|decision"))
    .withColumn("mentions_field", F.instr(F.lower(F.col("ai_business_impact_explanation")), F.lower(F.col("field_name"))) > 0)
)
display(quality_df.select("entity_name", "field_name", "issue_type", "rejected_count", "mentions_field", "has_risk_language"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7) Required Narrative: Implementation Review
# MAGIC
# MAGIC ### Quarantine Tables and Issue Types
# MAGIC - The notebook logs which quarantine tables were discovered and used.
# MAGIC - The grouped output reveals issue types found per entity/field.
# MAGIC
# MAGIC ### Prompt Template and Context
# MAGIC - Prompt includes entity, field, issue type, rejected count, sample triggering values, and source table context.
# MAGIC - This context forces specific, actionable explanations for finance auditors.
# MAGIC
# MAGIC ### Two Explanation Review
# MAGIC - Use the validation output table above to inspect two generated explanations.
# MAGIC - Confirm each mentions field + issue + business risk.
# MAGIC - If vague, tighten prompt constraints (for example: require field and count in sentence 1, risk in sentence 3).
