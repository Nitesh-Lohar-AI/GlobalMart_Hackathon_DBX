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
#  MATERIALIZED VIEWS (Business Failures #1–3b)
# ===========================================================================

# ---------------------------------------------------------------------------
# mv_monthly_revenue_by_region  — Business Failure #1
# ---------------------------------------------------------------------------
@dp.materialized_view(
    name=f"{GOLD}.mv_monthly_revenue_by_region",
    comment="BF#1 — Monthly revenue by region."
)
def mv_monthly_revenue_by_region():
    return spark.sql(f"""
        SELECT
            d.year,
            d.month_num,
            d.month_name,
            d.quarter,
            r.region_name,
            r.country,
            COUNT(DISTINCT fs.order_id)                             AS order_count,
            ROUND(SUM(fs.sales_amount), 2)                          AS total_sales_amount,
            SUM(fs.quantity)                                        AS total_quantity_sold,
            ROUND(SUM(fs.profit), 2)                                AS total_profit,
            ROUND(SUM(fs.discount), 2)                              AS total_discount,
            ROUND(
                SUM(fs.sales_amount) / NULLIF(COUNT(DISTINCT fs.order_id), 0)
            , 2)                                                    AS avg_order_value
        FROM      {GOLD}.fact_sales   fs
        JOIN      {GOLD}.dim_date     d   ON fs.date_key   = d.date_key
        JOIN      {GOLD}.dim_region   r   ON fs.region_name = r.region_name
        GROUP BY  d.year, d.month_num, d.month_name, d.quarter, r.region_name, r.country
        ORDER BY  d.year, d.month_num, r.region_name
    """)

# COMMAND ----------

# ---------------------------------------------------------------------------
# mv_customer_return_history  — Business Failure #2
# ---------------------------------------------------------------------------
@dp.materialized_view(
    name=f"{GOLD}.mv_customer_return_history",
    comment="BF#2 — Return count + refund value per customer."
)
def mv_customer_return_history():
    return spark.sql(f"""
        SELECT
            fr.customer_id,
            c.customer_name,
            c.customer_email,
            c.segment,
            c.region,
            c.country,
            COUNT(fr.order_id)                                      AS total_returns,
            ROUND(SUM(fr.refund_amount), 2)                         AS total_refund_value,
            ROUND(AVG(fr.refund_amount), 2)                         AS avg_refund_per_return,
            ROUND(AVG(fr.days_to_return), 1)                        AS avg_days_to_return,
            ROUND(
                SUM(CASE WHEN fr.is_approved = TRUE THEN 1 ELSE 0 END) * 100.0
                / NULLIF(COUNT(fr.order_id), 0)
            , 2)                                                    AS approval_rate_pct,
            MAX(TO_DATE(fr.return_date))                            AS last_return_date
        FROM      {GOLD}.fact_returns  fr
        JOIN      {GOLD}.dim_customer  c   ON fr.customer_id = c.customer_id
        GROUP BY  fr.customer_id, c.customer_name, c.customer_email, c.segment, c.region, c.country
        ORDER BY  total_returns DESC
    """)

# COMMAND ----------

# ---------------------------------------------------------------------------
# mv_vendor_return_rate  — Business Failure #3a
# ---------------------------------------------------------------------------
@dp.materialized_view(
    name=f"{GOLD}.mv_vendor_return_rate",
    comment="BF#3a — Return rate by vendor."
)
def mv_vendor_return_rate():
    return spark.sql(f"""
        WITH sales_by_vendor AS (
            SELECT vendor_id,
                   COUNT(DISTINCT order_id)     AS total_orders_sold,
                   ROUND(SUM(sales_amount), 2)  AS total_revenue
            FROM  {GOLD}.fact_sales
            GROUP BY vendor_id
        ),
        returns_by_vendor AS (
            SELECT vendor_id,
                   COUNT(order_id)              AS total_returns,
                   ROUND(SUM(refund_amount), 2) AS total_refund_value,
                   ROUND(AVG(days_to_return),1) AS avg_days_to_return
            FROM  {GOLD}.fact_returns
            GROUP BY vendor_id
        )
        SELECT
            v.vendor_id,
            v.vendor_name,
            COALESCE(s.total_orders_sold,  0)   AS total_orders_sold,
            COALESCE(s.total_revenue,      0)   AS total_revenue,
            COALESCE(r.total_returns,      0)   AS total_returns,
            COALESCE(r.total_refund_value, 0)   AS total_refund_value,
            ROUND(
                COALESCE(r.total_returns, 0) * 100.0
                / NULLIF(COALESCE(s.total_orders_sold, 0), 0)
            , 2)                                AS return_rate_pct,
            COALESCE(r.avg_days_to_return, 0)   AS avg_days_to_return
        FROM       {GOLD}.dim_vendor   v
        LEFT JOIN  sales_by_vendor               s  ON v.vendor_id = s.vendor_id
        LEFT JOIN  returns_by_vendor             r  ON v.vendor_id = r.vendor_id
        ORDER BY   return_rate_pct DESC
    """)

# COMMAND ----------

# ---------------------------------------------------------------------------
# mv_slow_moving_products  — Business Failure #3b
# NOTE: Product-level return data not available (fact_returns has no product_id).
#       Returns are tracked at order level only.
# ---------------------------------------------------------------------------
@dp.materialized_view(
    name=f"{GOLD}.mv_slow_moving_products",
    comment="BF#3b — Slow-moving products by region. Flag: <5 units OR 90+ days no sale."
)
def mv_slow_moving_products():
    return spark.sql(f"""
        WITH product_sales AS (
            SELECT
                fs.product_id,
                fs.region_name,
                COUNT(DISTINCT fs.order_id)     AS total_orders,
                SUM(fs.quantity)                AS total_units_sold,
                ROUND(SUM(fs.sales_amount), 2)  AS total_revenue,
                MAX(d.full_date)                AS last_sale_date
            FROM  {GOLD}.fact_sales  fs
            JOIN  {GOLD}.dim_date    d   ON fs.date_key = d.date_key
            GROUP BY fs.product_id, fs.region_name
        )
        SELECT
            p.product_id,
            p.product_name,
            p.category,
            p.sub_category,
            p.brand,
            r.region_name,
            COALESCE(ps.total_orders,     0)    AS total_orders,
            COALESCE(ps.total_units_sold, 0)    AS total_units_sold,
            COALESCE(ps.total_revenue,    0)    AS total_revenue,
            ps.last_sale_date,
            DATEDIFF(CURRENT_DATE(), ps.last_sale_date) AS days_since_last_sale,
            CASE
                WHEN COALESCE(ps.total_units_sold, 0) < 5
                  OR DATEDIFF(CURRENT_DATE(), ps.last_sale_date) > 90
                THEN TRUE ELSE FALSE
            END                                 AS is_slow_mover
        FROM       {GOLD}.dim_product   p
        CROSS JOIN {GOLD}.dim_region    r
        LEFT JOIN  product_sales                  ps
               ON  p.product_id = ps.product_id AND r.region_name = ps.region_name
        WHERE COALESCE(ps.total_orders, 0) > 0
        ORDER BY is_slow_mover DESC, days_since_last_sale DESC
    """)
