# GlobalMart Hackathon — Databricks Intelligence Platform

**A production-grade, end-to-end data and AI platform built on Databricks, demonstrating a fully automated retail analytics pipeline from raw multi-region data ingestion through executive-level generative AI insights.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Architecture](#architecture)
4. [Data Pipeline — ETL Layer](#data-pipeline--etl-layer)
   - [Initial Setup](#initial-setup)
   - [Bronze Layer — Auto Loader Ingestion](#bronze-layer--auto-loader-ingestion)
   - [Silver Layer — Standardization and Data Quality](#silver-layer--standardization-and-data-quality)
   - [Gold Layer — Dimensional Model and Materialized Views](#gold-layer--dimensional-model-and-materialized-views)
5. [Generative AI Use Cases](#generative-ai-use-cases)
   - [UC1 — AI Data Quality Reporter](#uc1--ai-data-quality-reporter)
   - [UC2 — Returns Fraud Investigator](#uc2--returns-fraud-investigator)
   - [UC3 — Product Intelligence Assistant (RAG)](#uc3--product-intelligence-assistant-rag)
   - [UC4 — AI Revenue and Inventory Intelligence](#uc4--ai-revenue-and-inventory-intelligence)
6. [Technology Stack](#technology-stack)
7. [Deployment and Configuration](#deployment-and-configuration)
8. [Catalog and Schema Layout](#catalog-and-schema-layout)
9. [Team](#team)

---

## Project Overview

GlobalMart is a multi-region retail enterprise operating across six geographic zones, each contributing structured data in inconsistent formats, schemas, and encodings. This project addresses the end-to-end challenge of ingesting, standardizing, and activating that data — culminating in four applied generative AI use cases that deliver measurable business value across finance, fraud operations, product intelligence, and executive leadership.

The platform is built entirely on **Databricks**, using Delta Live Tables for pipeline orchestration, Unity Catalog for governance, and the Databricks Model Serving endpoint to power LLM inference at scale — without any external LLM provider dependency.

The architecture is organized as a Medallion Lakehouse (Bronze / Silver / Gold) with a clean separation between the ETL pipeline and the GenAI application layer. All generative outputs are written back as queryable Delta tables in the Gold schema, making AI-generated insights first-class data assets available to downstream BI tools.

---

## Repository Structure

```
GlobalMart_Hackathon_DBX/
│
├── GlobalMart_Technical_Documentation.docx      # Full technical documentation
│
└── globalmart_hackathon_dbx_project/
    ├── databricks.yml                            # Databricks Asset Bundle definition
    ├── pyproject.toml                            # Python project dependencies
    │
    ├── resources/
    │   ├── globalmart_hackathon_dbx_project_etl.pipeline.yml   # DLT pipeline config
    │   └── sample_job.job.yml                                   # Sample job definition
    │
    └── src/
        ├── globalmart_hackathon_dbx_project_etl/
        │   ├── Intial_Setup/
        │   │   └── SETUP_NOTEBOOK.py                            # Catalog and volume setup
        │   └── transformations/
        │       └── Data_Layers/
        │           ├── Bronze_Layer/
        │           │   └── bronze_layer.py                      # Auto Loader ingestion (6 entities)
        │           ├── Silver_Layer/
        │           │   ├── silver_customer_cleaning.py
        │           │   ├── silver_ordes_cleaning.py
        │           │   ├── silver_products_cleaning.py
        │           │   ├── silver_returns_cleaning.py
        │           │   ├── silver_transactions_cleaning.py
        │           │   └── silver_vendors_cleaning.py
        │           └── Gold_Layer/
        │               ├── Gold_Layer.py                        # Dimensions and facts
        │               └── Gold_Matrix_MV.py                    # Business Failure MVs
        │
        └── globalmart_hackathon_dbx_project_gen_ai/
            ├── uc1_dq_reporter.py                               # UC1: DQ Audit Reporter
            ├── uc2_returns_fraud_investigator.py                # UC2: Fraud Investigator
            ├── uc3_product_intelligence_rag.py                  # UC3: RAG Assistant
            └── uc4_ai_revenue_inventory_intelligence.py         # UC4: Executive Intelligence
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SOURCE DATA (Cloud Volumes)                         │
│   6 Regions · CSV / JSON · Schema Variants · 6 Entities                     │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │  Auto Loader (cloudFiles)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BRONZE LAYER (Delta)                              │
│   Raw ingestion · Schema evolution · Audit metadata · No transformation     │
│   Tables: bronze_customers, bronze_orders, bronze_transactions,             │
│           bronze_returns, bronze_products, bronze_vendors                   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │  DLT Expectations · Quarantine Routing
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SILVER LAYER (Delta)                              │
│   Schema unification · Segment / region normalization · DQ rules            │
│   Clean tables + Quarantine tables (per entity)                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │  Materialized Views · Streaming Facts
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GOLD LAYER (Delta)                               │
│   Star Schema: dim_date, dim_customer, dim_vendor, dim_product, dim_region  │
│   Facts: fact_sales, fact_returns                                           │
│   MVs: mv_monthly_revenue_by_region, mv_vendor_return_rate,                 │
│        mv_customer_return_history, mv_slow_moving_products                  │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │  OpenAI SDK · Databricks Model Serving
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GENERATIVE AI APPLICATION LAYER                        │
│   UC1: DQ Audit Report     → gold.dq_audit_report                           │
│   UC2: Fraud Investigation → gold.flagged_return_customers                  │
│   UC3: RAG Assistant       → gold.rag_query_history                         │
│   UC4: Executive Insights  → gold.ai_business_insights                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

All pipeline stages are orchestrated through **Databricks Delta Live Tables (DLT)** running in serverless mode, governed by **Unity Catalog**, and deployed via **Databricks Asset Bundles** with explicit dev and prod targets.

---

## Catalog and Schema Layout

All data assets reside in the `dbx_hck_glbl_mart` Unity Catalog.

```
dbx_hck_glbl_mart/
├── bronze/
│   ├── bronze_customers
│   ├── bronze_orders
│   ├── bronze_transactions
│   ├── bronze_returns
│   ├── bronze_products
│   └── bronze_vendors
│
├── silver/
│   ├── silver_customers
│   ├── silver_customers_quaruntine
│   ├── silver_orders_clean
│   ├── silver_orders_quaruntine
│   ├── silver_transactions
│   ├── silver_transactions_quarantine
│   ├── silver_products
│   ├── silver_products_quarantine
│   ├── silver_returns
│   ├── silver_returns_quarantine
│   ├── silver_vendors
│   └── silver_vendors_quarantine
│
└── gold/
    ├── dim_date
    ├── dim_customer
    ├── dim_product
    ├── dim_vendor
    ├── dim_region
    ├── fact_sales
    ├── fact_returns
    ├── mv_monthly_revenue_by_region
    ├── mv_customer_return_history
    ├── mv_vendor_return_rate
    ├── mv_slow_moving_products
    ├── dq_audit_report              -- UC1 GenAI output
    ├── flagged_return_customers     -- UC2 GenAI output
    ├── rag_query_history            -- UC3 GenAI output
    └── ai_business_insights         -- UC4 GenAI output
```

---

## Data Pipeline — ETL Layer

### Initial Setup

The `SETUP_NOTEBOOK.py` provisions the Unity Catalog hierarchy and the cloud storage volume required by the pipeline before any data is ingested. This ensures idempotent environment initialization across dev and prod workspaces.

---

### Bronze Layer — Auto Loader Ingestion

**Notebook:** `transformations/Data_Layers/Bronze_Layer/bronze_layer.py`

The Bronze layer uses **Databricks Auto Loader** (`cloudFiles` format) to continuously ingest data from a Unity Catalog Volume (`/Volumes/dbx_hck_glbl_mart/bronze/raw_landing`). Six entities are ingested as streaming Delta tables.

**Key design decisions:**

**Schema evolution with `addNewColumns` mode.** When Region 6 introduced a new "Birth Date" column mid-project, the pipeline absorbed it without code changes. Prior records receive NULL; new records carry the value.

**Schema hints for multi-schema sources.** Each entity defines schema hints covering every known column variant across all six regions. Auto Loader unifies case-insensitive column matches and adds new variants as separate columns.

**Audit metadata injection.** Every record receives `_source_file`, `_load_timestamp`, and `_region` (extracted from the file path) for full data lineage.

**`recursiveFileLookup` with `pathGlobFilter`.** A single Auto Loader stream discovers files in the root volume and all Region subdirectories, enabling flexible source layouts.

| Entity | Format | Schema Complexity |
|---|---|---|
| Customers | CSV | 4 ID variants, 2 email variants, city/state swap in Region 4 |
| Orders | CSV | Date strings in 2 formats across regions |
| Transactions | CSV | `discount` as string ("40%") in Region 4, missing `profit` |
| Returns | JSON | Two files with completely different column names |
| Products | JSON | UPC stored as STRING to prevent scientific notation loss |
| Vendors | CSV | Simple 2-column schema |

---

### Silver Layer — Standardization and Data Quality

**Notebooks:** `transformations/Data_Layers/Silver_Layer/`

The Silver layer transforms six raw Bronze entities into clean, validated Delta tables using **Spark Declarative Pipelines (DLT)**. Each entity follows a three-step pattern.

**Step 1 — Standardization view.** A `@dp.temporary_view()` resolves all schema variants using `coalesce()`, normalizes categorical values, corrects known regional data issues (e.g., city/state swap in Region 4 customers), and casts data types.

**Step 2 — Quarantine table.** A `@dp.table()` captures every record that fails DQ rules into a dedicated quarantine table. Each quarantine row includes an `error_reason` column identifying the failing rule, enabling the UC1 GenAI use case downstream.

**Step 3 — Clean table with DLT Expectations.** A `@dp.expect_all_or_drop()` decorator enforces business rules at the DLT engine level. Records failing any expectation are dropped from the clean Silver table and automatically routed to the quarantine table.

**Entities and key transformations:**

**Customers** — Consolidates four customer ID variants, two email and name variants, normalizes segment values (Consumer / Corporate / Home Office with typo correction), standardizes region codes (E → East, W → West, etc.), and corrects the city/state column swap present in Region 4 source files.

**Orders** — Unifies order date formats across regions (`MM/DD/YYYY` and `YYYY-MM-DD`), standardizes ship mode and order status values.

**Transactions** — Parses percentage discount strings (e.g., "40%") into numeric values, casts sales and profit to correct numeric types, handles NULL profit from Region 4.

**Returns** — Reconciles two completely different JSON schemas into a unified schema (`order_id` / `OrderId`, `return_reason` / `reason`, `return_date` / `date_of_return`, `refund_amount` / `amount`, `return_status` / `status`).

**Products** — Retains UPC as STRING throughout to prevent floating-point precision loss from scientific notation values (e.g., `6.40E+11`), standardizes category hierarchies.

**Vendors** — Simple deduplication and vendor ID validation.

---

### Gold Layer — Dimensional Model and Materialized Views

**Notebooks:** `transformations/Data_Layers/Gold_Layer/Gold_Layer.py` and `Gold_Matrix_MV.py`

The Gold layer implements a **star schema** dimensional model optimized for analytics. Dimensions use `@dp.materialized_view()` for full-refresh on each pipeline run (appropriate for slowly-changing dimensions). Facts use `dp.create_streaming_table()` with `@dp.append_flow()` for incremental processing driven by Silver streams.

**Dimension Tables:**

`dim_date` — Generated date spine from 2010 to 2050. Includes day of week, week number, month name, quarter, year, and weekend flag. Role-played twice in `fact_returns` (return date and order date).

`dim_customer` — Distinct customer records with full demographic profile from `silver_customers`.

`dim_product` — Product catalog enriched with Gold-derived `category` and `sub_category` columns, split from the raw `categories` field using hierarchical parsing.

`dim_vendor` — Seven vendor records (VEN01–VEN07) from `silver_vendors`.

`dim_region` — Distinct region / country combinations derived from `silver_customers`.

**Fact Tables:**

`fact_sales` — Grain: one row per order + product combination. Streaming source: `silver_transactions`. Broadcast-joined to `silver_orders_clean` and `silver_customers` for denormalization. Measures: `sales_amount`, `quantity`, `discount`, `profit`, `payment_type`, `payment_installments`.

`fact_returns` — Grain: one return per order. Streaming source: `silver_returns`. Gold-derived columns: `is_approved` (parsed from `return_status`) and `days_to_return` (computed from return date and order purchase date). `dim_date` is role-played twice — `return_date_key` and `order_date_key`.

**Business Failure Materialized Views:**

`mv_monthly_revenue_by_region` — Monthly revenue, order count, quantity sold, profit, discount, and average order value joined across `fact_sales`, `dim_date`, and `dim_region`.

`mv_customer_return_history` — Return frequency, total refund value, average refund, average days to return, and approval rate per customer.

`mv_vendor_return_rate` — Return rate percentage, total returns, total revenue, and average days to return per vendor. Computed with a CTE pattern to avoid division-by-zero.

`mv_slow_moving_products` — Cross-join of `dim_product` and `dim_region` against `fact_sales` to identify products with fewer than 5 units sold or more than 90 days since last sale in a given region.

---

## Generative AI Use Cases

All four use cases use the **Databricks OpenAI-compatible serving endpoint** (`databricks-gpt-oss-20b`) via the OpenAI Python SDK, with the workspace token and URL dynamically resolved at runtime. No external API keys or third-party LLM providers are required. All outputs are written to queryable Delta tables in the Gold schema.

---

### UC1 — AI Data Quality Reporter

**Notebook:** `uc1_dq_reporter.py`  
**Output Table:** `dbx_hck_glbl_mart.gold.dq_audit_report`

**Business problem.** Silver quarantine tables capture records that failed DQ validation, but the raw quarantine data is technical in nature and inaccessible to finance auditors who need to understand business impact.

**Solution.** The notebook discovers all quarantine tables across the Silver schema (handling both `quarantine` and `quaruntine` spelling variants), normalizes records from all six entities into a unified schema, groups rejections by entity / field / issue type, and sends each group to the LLM with a structured prompt requesting a finance-friendly, plain-English paragraph that explains the issue, why records were rejected, and which business reports or audit figures are at risk.

**Pipeline stages:**

1. Dynamic quarantine table discovery using `SHOW TABLES` with name-pattern matching.
2. Schema normalization across heterogeneous quarantine schemas using `first_existing_col()` with ordered candidate column lists.
3. Aggregation: `countDistinct` on `record_id` and `collect_list` of sample trigger values per issue group.
4. LLM prompt construction with entity, field, issue type, count, sample values, and source table context.
5. `databricks-gpt-oss-20b` inference with temperature 0.2 for consistent, professional output.
6. Response parsing handling both JSON-wrapped (`{"type": "text", ...}`) and raw string content formats.
7. Output written to `gold.dq_audit_report` as a Delta table with overwrite mode.

**Output schema:** `entity_name`, `field_name`, `issue_type`, `rejected_count`, `sample_trigger_values`, `source_tables`, `ai_business_impact_explanation`, `generated_at`.

**Business value.** Finance and audit teams receive a structured, queryable report that translates technical DQ failures into language they can act on — without requiring SQL or data engineering expertise.

---

### UC2 — Returns Fraud Investigator

**Notebook:** `uc2_returns_fraud_investigator.py`  
**Output Table:** `dbx_hck_glbl_mart.gold.flagged_return_customers`

**Business problem.** Identifying customers who abuse the returns process requires combining multiple behavioral signals into a coherent risk score, then providing investigation teams with actionable briefs — not just a score.

**Solution.** The notebook builds behavioral risk profiles from `fact_returns` and `dim_customer`, applies a weighted five-rule anomaly scoring model, flags customers above a configurable threshold, and generates a structured investigation brief for each flagged customer using the LLM.

**Pipeline stages:**

1. Feature engineering from `fact_returns`: total returns, total return value, average return value, distinct regions used for returns, count and ratio of negative days-to-return (return logged before order date), non-approved return ratio, and burst window counts (7-day and 30-day).
2. Left-join enrichment with `dim_customer` for name, region, and tier context.
3. Weighted anomaly scoring across five rules:

| Rule | Weight | Threshold |
|---|---|---|
| High return frequency | 20 | >= 8 total returns |
| High return value | 25 | >= $1,000 total |
| Negative days to return | 20 | >= 15% of returns |
| Multi-region returns | 15 | >= 3 distinct regions |
| High non-approved ratio | 20 | >= 50% non-approved |

4. Maximum total score: 100. Flag threshold: 65 (configurable). Threshold sensitivity analysis comparing flagged counts at the primary and lower thresholds.
5. LLM prompt requests a strictly formatted investigation brief: three bullets each for suspicious patterns, possible innocent explanations, and first verification actions — with concrete numeric values mandated.
6. Output written to `gold.flagged_return_customers` as a Delta table.

**Output schema:** full customer profile metrics, `anomaly_score`, `rules_violated`, `ai_investigation_brief`, `generated_at`.

**Business value.** Returns operations managers receive a prioritized, actionable investigation queue with pre-written briefs that include specific numeric evidence, saving hours of manual profile review per case.

---

### UC3 — Product Intelligence Assistant (RAG)

**Notebook:** `uc3_product_intelligence_rag.py`  
**Output Table:** `dbx_hck_glbl_mart.gold.rag_query_history`

**Business problem.** Product and category managers need to query product performance, vendor health, and inventory positioning using natural language — without writing SQL or waiting for a BI report to be built.

**Solution.** A local-embedding Retrieval-Augmented Generation (RAG) system built entirely within the Databricks environment. Product and vendor data from Gold tables is converted to natural-language documents, embedded using a sentence transformer model, indexed in a FAISS vector store, and made queryable through an LLM that answers grounded strictly on retrieved context.

**Pipeline stages:**

1. Data loading from five Gold tables: `dim_product`, `dim_vendor`, `mv_monthly_revenue_by_region`, `mv_vendor_return_rate`, `mv_slow_moving_products`.
2. Schema-aligned normalization and enrichment: products enriched with slow-mover flags, days since last sale, and regional revenue context; vendors enriched with return rate metrics.
3. Document generation: each product and vendor is serialized into a structured natural-language text document (e.g., "Product intelligence profile. Product name: ... Category: ... Slow mover flag: ... Region total sales context: ...").
4. Embedding with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, normalized).
5. FAISS `IndexFlatIP` (inner product) index built in-memory across all documents.
6. Retrieval: cosine similarity search returning the top-K most relevant documents per query.
7. Grounded generation: retrieved documents passed to `databricks-gpt-oss-20b` with strict rules to answer only from context and cite concrete product/vendor names.
8. MLflow autologging via `mlflow.openai.autolog()` for experiment tracking.
9. Query and answer pairs logged to `gold.rag_query_history` for audit and evaluation.

**Representative queries answered:**
- "Which products are marked as slow movers in the West region?"
- "Which vendor has the highest return rate percentage?"
- "Show product examples with very high days since last sale."
- "What does the retrieved context say about average order value in the East region?"

**Output schema:** `question`, `answer`, `retrieved_documents` (JSON), `retrieved_metadata` (JSON), `top_k`, `generated_at`.

**Business value.** Product managers can query the full product and vendor intelligence corpus in natural language, with answers grounded in live Gold data and every response logged for governance and audit.

---

### UC4 — AI Revenue and Inventory Intelligence

**Notebook:** `uc4_ai_revenue_inventory_intelligence.py`  
**Output Table:** `globalmart.gold.ai_business_insights`

**Business problem.** Executives need concise, data-grounded summaries of key performance signals across revenue, vendor risk, and inventory health — on demand, without navigating dashboards or waiting for analyst reports.

**Solution.** The notebook reads three Gold materialized views, computes aggregated KPI payloads for each business domain, passes the KPI JSON (not raw rows) to the LLM for executive summary generation, demonstrates Databricks `ai_query()` SQL function integration, and writes all outputs to a queryable Gold table.

**Pipeline stages:**

1. Data profiling of all three MV sources (`mv_monthly_revenue_by_region`, `mv_vendor_return_rate`, `mv_slow_moving_products`) with null counts and schema validation.
2. KPI payload computation for three insight domains:

   **Revenue performance:** total revenue, average revenue per region, top-3 and bottom-3 regions by revenue, region count.

   **Vendor return rate:** average return rate across all vendors, count of vendors with return rate >= 20%, top-5 vendors by return rate with total returns.

   **Slow-moving inventory:** total slow-moving product count, total slow-moving inventory value, top regions by slow inventory value and product count.

3. Executive summary generation: each KPI payload is passed to `databricks-gpt-oss-20b` with a prompt requesting a 4-6 sentence summary stating the most important signals, interpreting business risk or opportunity, and recommending immediate leadership focus areas. The LLM operates on aggregated KPIs only — no raw row data is passed.

4. `ai_query()` SQL demonstrations: two native Databricks SQL cells show how the same LLM inference can be driven directly from SQL using `ai_query('databricks-gpt-oss-20b', concat(...))` — one for regional revenue assessment and one for vendor return-rate risk.

5. Output written to `globalmart.gold.ai_business_insights` with `insight_type`, `executive_summary`, `kpi_payload_json`, and `generated_at`.

**Business value.** Leadership receives concise, fact-grounded executive summaries on revenue performance, vendor risk concentration, and slow inventory exposure — generated from live data and available as a queryable Delta table for integration into any BI tool or dashboard.

---

## Technology Stack

| Component | Technology |
|---|---|
| Platform | Databricks (Serverless) |
| Pipeline Orchestration | Lackflow Declarative Pipelines (Spark Declarative Pipelines) |
| Streaming Ingestion | Databricks Auto Loader (cloudFiles) |
| Storage Format | Delta Lake |
| Data Governance | Unity Catalog |
| Deployment | Databricks Asset Bundles |
| LLM Inference | `databricks-gpt-oss-20b` via Databricks Model Serving |
| LLM Client | OpenAI Python SDK (Databricks-compatible endpoint) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS (`IndexFlatIP`, in-memory) |
| Experiment Tracking | MLflow (OpenAI autologging) |
| SQL AI Integration | Databricks `ai_query()` |
| Language | Python 3, PySpark, Spark SQL |

---

## Deployment and Configuration

The project is managed as a **Databricks Asset Bundle** defined in `databricks.yml`.

**Workspace:** `https://dbc-f642ffc2-498c.cloud.databricks.com`

**Dev target** (default) — Deploys with `mode: development`. All deployed resources are prefixed with `[dev <username>]`. Job schedules and triggers are paused. Variables: `catalog = dbx_hck_glbl_mart`, `schema = dev`.

**Prod target** — Deploys to `/Workspace/Users/rahul.vaghasia@kenexai.com/.bundle/`. Variables: `catalog = dbx_hck_glbl_mart`, `schema = prod`. Three team members have `CAN_MANAGE` permission.

**DLT Pipeline configuration** (`globalmart_hackathon_dbx_project_etl.pipeline.yml`):

```yaml
pipelines:
  globalmart_hackathon_dbx_project_etl:
    catalog: ${var.catalog}
    schema: ${var.schema}
    serverless: true
    libraries:
      - glob:
          include: ../src/globalmart_hackathon_dbx_project_etl/transformations/**
```

All transformation notebooks under the `transformations/` directory are included automatically via glob pattern. The pipeline runs in serverless mode with no cluster management required.

---



## Team

| Name | Role |
|---|---|
| Nitesh Lohar | Data Engineering and Data Modeling |
| Rahul Vaghasia | Pipeline Architecture and Data Engineering |
| Mukesh Goswami | GenAI Development |

Organization: **KenexAI**
