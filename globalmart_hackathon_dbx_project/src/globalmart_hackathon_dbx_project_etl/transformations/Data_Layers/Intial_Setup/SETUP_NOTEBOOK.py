# Databricks notebook source
# MAGIC %sql
# MAGIC -- catalog creation
# MAGIC create catalog if not exists `dbx_hck_glbl_mart`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- creating schema inside catalog
# MAGIC create schema if not exists dbx_hck_glbl_mart.bronze;
# MAGIC create schema if not exists dbx_hck_glbl_mart.silver;
# MAGIC create schema if not exists dbx_hck_glbl_mart.gold; 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- intializing volume in bronze
# MAGIC create volume if not exists dbx_hck_glbl_mart.bronze.raw_landing;
