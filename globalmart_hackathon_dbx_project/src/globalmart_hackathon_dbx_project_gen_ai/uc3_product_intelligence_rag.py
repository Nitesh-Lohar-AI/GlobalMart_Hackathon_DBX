# Databricks notebook source
# MAGIC %md
# MAGIC # UC3 Part A: Product Intelligence Assistant (RAG)
# MAGIC
# MAGIC This notebook builds a local-embedding RAG system over GlobalMart Gold product/vendor data, answers grounded questions, and logs outputs to `globalmart.gold.rag_query_history`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Setup and Dependencies

# COMMAND ----------

# Optional install cells for Databricks runtimes that do not include these packages.
%pip install sentence-transformers faiss-cpu
dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %pip install -U mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install openai

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from openai import OpenAI
import json
import re
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from pyspark.sql import functions as F
from pyspark.sql import types as T

MODEL_NAME = "databricks-gpt-oss-20b"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CATALOG = "dbx_hck_glbl_mart"
SCHEMA = "gold"

DIM_PRODUCTS = f"{CATALOG}.{SCHEMA}.dim_product"
DIM_VENDORS = f"{CATALOG}.{SCHEMA}.dim_vendor"
MV_REVENUE_BY_REGION = f"{CATALOG}.{SCHEMA}.mv_monthly_revenue_by_region"
MV_RETURN_RATE_BY_VENDOR = f"{CATALOG}.{SCHEMA}.mv_vendor_return_rate"
MV_SLOW_MOVING_PRODUCTS = f"{CATALOG}.{SCHEMA}.mv_slow_moving_products"

RAG_LOG_TABLE = f"{CATALOG}.{SCHEMA}.rag_query_history"
DEFAULT_TOP_K = 5


# COMMAND ----------

import mlflow
mlflow.openai.autolog()

# COMMAND ----------

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_ws_raw = spark.conf.get("spark.databricks.workspaceUrl")
_ws_url = _ws_raw.replace("https://", "").replace("http://", "").strip("/")

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"https://{_ws_url}/serving-endpoints"
)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print(f"LLM model: {MODEL_NAME}")
print(f"Embedding model: {EMBED_MODEL_NAME}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Read Gold Data and Build Documents

# COMMAND ----------

products_df = spark.table(DIM_PRODUCTS)
vendors_df = spark.table(DIM_VENDORS)

rev_region_df = spark.table(MV_REVENUE_BY_REGION)
rr_vendor_df = spark.table(MV_RETURN_RATE_BY_VENDOR)
slow_moving_df = spark.table(MV_SLOW_MOVING_PRODUCTS)

print("dim_product columns:", products_df.columns)
print("dim_vendor columns:", vendors_df.columns)
print("mv_monthly_revenue_by_region columns:", rev_region_df.columns)
print("mv_vendor_return_rate columns:", rr_vendor_df.columns)
print("mv_slow_moving_products columns:", slow_moving_df.columns)


# COMMAND ----------

# Schema-aligned normalization using known columns in Gold tables.

products_norm = products_df.select(
    F.col("product_id").cast("string").alias("product_id"),
    F.col("product_name").cast("string").alias("product_name"),
    F.col("category").cast("string").alias("category"),
    F.col("sub_category").cast("string").alias("sub_category"),
    F.col("brand").cast("string").alias("brand"),
    F.col("manufacturer").cast("string").alias("manufacturer"),
)

vendors_norm = vendors_df.select(
    F.col("vendor_id").cast("string").alias("vendor_id"),
    F.col("vendor_name").cast("string").alias("vendor_name"),
)

rr_norm = rr_vendor_df.select(
    F.col("vendor_id").cast("string").alias("vendor_id"),
    F.col("vendor_name").cast("string").alias("vendor_name"),
    F.col("return_rate_pct").cast("double").alias("vendor_return_rate_pct"),
    F.col("avg_days_to_return").cast("double").alias("vendor_avg_days_to_return"),
    F.col("total_returns").cast("long").alias("vendor_total_returns"),
    F.col("total_revenue").cast("double").alias("vendor_total_revenue"),
)

# Region-level revenue summary from monthly MV
rev_norm = (
    rev_region_df
    .groupBy("region_name")
    .agg(
        F.round(F.sum(F.col("total_sales_amount")), 2).alias("region_total_sales"),
        F.round(F.avg(F.col("avg_order_value")), 2).alias("region_avg_order_value"),
        F.countDistinct(F.concat_ws("-", F.col("year").cast("string"), F.col("month_num").cast("string"))).alias("months_covered"),
    )
    .withColumnRenamed("region_name", "region")
)

# Product-level slow-mover summary (many rows per product across regions)
slow_norm = (
    slow_moving_df
    .groupBy("product_id")
    .agg(
        F.max(F.col("is_slow_mover").cast("int")).cast("boolean").alias("is_slow_mover"),
        F.max(F.col("days_since_last_sale")).alias("max_days_since_last_sale"),
        F.round(F.sum(F.coalesce(F.col("total_revenue"), F.lit(0.0))), 2).alias("slow_mover_total_revenue"),
        F.array_join(F.slice(F.array_sort(F.collect_set(F.col("region_name"))), 1, 3), ", ").alias("slow_regions_sample"),
    )
)

# Choose a representative region per product (highest days_since_last_sale row)
slow_region_pick = (
    slow_moving_df
    .select("product_id", "region_name", "days_since_last_sale")
    .where(F.col("product_id").isNotNull())
    .orderBy(F.col("days_since_last_sale").desc_nulls_last())
    .dropDuplicates(["product_id"])
    .select(
        F.col("product_id").cast("string").alias("product_id"),
        F.col("region_name").cast("string").alias("region"),
    )
)

# Enrich products with slow-mover and region revenue context.
product_enriched = (
    products_norm.alias("p")
    .join(slow_norm.alias("s"), on="product_id", how="left")
    .join(slow_region_pick.alias("sr"), on="product_id", how="left")
    .join(rev_norm.alias("rv"), on="region", how="left")
)

# Enrich vendor docs with return-rate MV metrics.
vendor_enriched = (
    vendors_norm.alias("v")
    .join(rr_norm.alias("rr"), on=["vendor_id", "vendor_name"], how="left")
)

display(product_enriched.limit(20))
display(vendor_enriched.limit(20))


# COMMAND ----------

product_docs_df = product_enriched.select(
    F.concat(F.lit("product::"), F.col("product_id")).alias("doc_id"),
    F.lit("product").alias("doc_type"),
    F.col("product_id"),
    F.lit(None).cast("string").alias("vendor_id"),
    F.coalesce(F.col("region"), F.lit("unknown")).alias("region"),
    F.concat_ws(
        " ",
        F.lit("Product intelligence profile."),
        F.concat(F.lit("Product name: "), F.coalesce(F.col("product_name"), F.lit("unknown")), F.lit(".")),
        F.concat(F.lit("Category: "), F.coalesce(F.col("category"), F.lit("unknown")), F.lit(".")),
        F.concat(F.lit("Sub-category: "), F.coalesce(F.col("sub_category"), F.lit("unknown")), F.lit(".")),
        F.concat(F.lit("Brand: "), F.coalesce(F.col("brand"), F.lit("unknown")), F.lit(".")),
        F.concat(F.lit("Manufacturer: "), F.coalesce(F.col("manufacturer"), F.lit("unknown")), F.lit(".")),
        F.concat(F.lit("Region context: "), F.coalesce(F.col("region"), F.lit("unknown")), F.lit(".")),
        F.concat(F.lit("Slow mover flag: "), F.coalesce(F.col("is_slow_mover").cast("string"), F.lit("false")), F.lit(".")),
        F.concat(F.lit("Max days since last sale: "), F.coalesce(F.col("max_days_since_last_sale").cast("string"), F.lit("n/a")), F.lit(".")),
        F.concat(F.lit("Slow-mover regions sample: "), F.coalesce(F.col("slow_regions_sample"), F.lit("n/a")), F.lit(".")),
        F.concat(F.lit("Slow-mover revenue sum: "), F.coalesce(F.col("slow_mover_total_revenue").cast("string"), F.lit("n/a")), F.lit(".")),
        F.concat(F.lit("Region total sales context: "), F.coalesce(F.col("region_total_sales").cast("string"), F.lit("n/a")), F.lit(".")),
        F.concat(F.lit("Region average order value context: "), F.coalesce(F.col("region_avg_order_value").cast("string"), F.lit("n/a")), F.lit("."))
    ).alias("doc_text")
)

vendor_docs_df = vendor_enriched.select(
    F.concat(F.lit("vendor::"), F.col("vendor_id")).alias("doc_id"),
    F.lit("vendor").alias("doc_type"),
    F.lit(None).cast("string").alias("product_id"),
    F.col("vendor_id"),
    F.lit("global").alias("region"),
    F.concat_ws(
        " ",
        F.lit("Vendor intelligence profile."),
        F.concat(F.lit("Vendor name: "), F.coalesce(F.col("vendor_name"), F.lit("unknown")), F.lit(".")),
        F.concat(F.lit("Vendor return rate pct: "), F.coalesce(F.col("vendor_return_rate_pct").cast("string"), F.lit("n/a")), F.lit(".")),
        F.concat(F.lit("Vendor average days to return: "), F.coalesce(F.col("vendor_avg_days_to_return").cast("string"), F.lit("n/a")), F.lit(".")),
        F.concat(F.lit("Vendor total returns: "), F.coalesce(F.col("vendor_total_returns").cast("string"), F.lit("n/a")), F.lit(".")),
        F.concat(F.lit("Vendor total revenue: "), F.coalesce(F.col("vendor_total_revenue").cast("string"), F.lit("n/a")), F.lit("."))
    ).alias("doc_text")
)

docs_df = product_docs_df.unionByName(vendor_docs_df)
display(docs_df.limit(20))
print(f"Total documents for index: {docs_df.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Local Embeddings and FAISS Index

# COMMAND ----------

docs = docs_df.select("doc_id", "doc_type", "product_id", "vendor_id", "region", "doc_text").collect()
doc_texts = [r["doc_text"] for r in docs]
doc_meta = [
    {
        "doc_id": r["doc_id"],
        "doc_type": r["doc_type"],
        "product_id": r["product_id"],
        "vendor_id": r["vendor_id"],
        "region": r["region"]
    }
    for r in docs
]

if not doc_texts:
    raise ValueError("No documents available for FAISS indexing.")

emb = embed_model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True)
emb = emb.astype("float32")

dimension = emb.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(emb)

print(f"FAISS index dimension: {dimension}")
print(f"Indexed documents: {index.ntotal}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4) Retrieval and Grounded Answer Generation

# COMMAND ----------

def extract_text_block(content):
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                txt = item.get("text", "")
                if txt and txt.strip():
                    return txt.strip()

    if isinstance(content, str):
        s = content.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("type") == "text":
                        txt = item.get("text", "")
                        if txt and txt.strip():
                            return txt.strip()
        except Exception:
            pass
        return re.sub(r"\s+", " ", s).strip()
    return ""

def retrieve_docs(question, top_k=DEFAULT_TOP_K):
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    rows = []
    for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
        if i < 0:
            continue
        rows.append({
            "rank": len(rows) + 1,
            "score": float(score),
            "doc_text": doc_texts[i],
            "metadata": doc_meta[i]
        })
    return rows

def answer_with_grounding(question, retrieved_rows):
    context = "\n\n".join([
        f"[Doc {r['rank']} | score={r['score']:.4f} | meta={json.dumps(r['metadata'])}]\n{r['doc_text']}"
        for r in retrieved_rows
    ])

    prompt = f"""
You are a product intelligence assistant for GlobalMart.

Question:
{question}

Retrieved context documents:
{context}

Rules:
1) Answer only from the retrieved context above.
2) If the answer is not clearly present, respond exactly: "The answer is not available in the retrieved documents.".
3) Cite concrete product/vendor names when available.
4) Keep response concise and factual.
""".strip()

    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You provide grounded answers from provided context only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = resp.choices[0].message.content
            text = extract_text_block(content)
            if text:
                return text
        except Exception:
            continue

    return "The answer is not available in the retrieved documents."

def ask_rag(question, top_k=DEFAULT_TOP_K):
    retrieved = retrieve_docs(question, top_k=top_k)
    answer = answer_with_grounding(question, retrieved)
    return {
        "question": question,
        "answer": answer,
        "retrieved": retrieved,
        "top_k": top_k,
        "generated_at": datetime.utcnow().isoformat()
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5) Query Logging to `dbx_hck_glbl_mart.gold.rag_query_history`

# COMMAND ----------

TEST_QUESTIONS = [
    "Which products are marked as slow movers in the West region?",
    "Which vendor has the highest return rate percentage?",
    "What does the retrieved context say about average order value in the East region?",
    "Show product examples with very high days since last sale.",
    "Which vendor appears to have the highest total returns in the available context?"
]

results = [ask_rag(q, top_k=5) for q in TEST_QUESTIONS]

for r in results:
    print("=" * 90)
    print("Question:", r["question"])
    print("Answer:", r["answer"])
    print("Retrieved docs:")
    for d in r["retrieved"]:
        print(f"  - rank={d['rank']} score={d['score']:.4f} meta={d['metadata']}")


# COMMAND ----------

log_rows = []
for r in results:
    retrieved_docs = [x["doc_text"] for x in r["retrieved"]]
    retrieved_meta = [x["metadata"] for x in r["retrieved"]]

    log_rows.append({
        "question": r["question"],
        "answer": r["answer"],
        "retrieved_documents": json.dumps(retrieved_docs),
        "retrieved_metadata": json.dumps(retrieved_meta),
        "top_k": int(r["top_k"]),
        "generated_at": r["generated_at"]
    })

log_schema = T.StructType([
    T.StructField("question", T.StringType(), True),
    T.StructField("answer", T.StringType(), True),
    T.StructField("retrieved_documents", T.StringType(), True),
    T.StructField("retrieved_metadata", T.StringType(), True),
    T.StructField("top_k", T.IntegerType(), True),
    T.StructField("generated_at", T.StringType(), True)
])

log_df = spark.createDataFrame(log_rows, schema=log_schema)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
(
    log_df
    .write
    .mode("append")
    .format("delta")
    .saveAsTable(RAG_LOG_TABLE)
)

print(f"Logged {log_df.count()} queries to {RAG_LOG_TABLE}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6) Validation Views

# COMMAND ----------

history_df = spark.table(RAG_LOG_TABLE)
print(f"rag_query_history row count: {history_df.count()}")
history_df.printSchema()
display(history_df.orderBy(F.desc("generated_at")).limit(20))


# COMMAND ----------

sample_df = history_df.orderBy(F.desc("generated_at")).select(
    "question", "answer", "retrieved_documents", "retrieved_metadata", "generated_at"
).limit(3)
display(sample_df)
