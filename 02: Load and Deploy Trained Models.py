# Databricks notebook source
# MAGIC %md
# MAGIC # Load and Deploy Models with MLflow and Python
# MAGIC The following notebook will help us register and deploy models trained in R. 

# COMMAND ----------

# MAGIC %md
# MAGIC Here we load the previously trained models in R and loading them in Python. Then, we can register and serve our models using [MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) and [MLflow Model Serving](https://docs.databricks.com/applications/mlflow/model-serving.html). 

# COMMAND ----------

# MAGIC %md
# MAGIC First, we make sure mlflow is up to date, feel free to skip if you are using the latest runtimes.

# COMMAND ----------

# MAGIC %pip install --upgrade mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC We'll set up some widgets to parametrize our notebook and accept user input

# COMMAND ----------

# Creates widgets 
dbutils.widgets.text("r_model_name","registered_model_name")
dbutils.widgets.text("r_model_version","1")

dbutils.widgets.text("python_model_name","model_name_to_register")

# COMMAND ----------

# Define variables
r_model_name = dbutils.widgets.get("r_model_name")
r_model_version = dbutils.widgets.get("r_model_version")

python_model_name = dbutils.widgets.get("python_model_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading dependencies:

# COMMAND ----------

import mlflow
import tensorflow as tf

from mlflow.client import MlflowClient
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load model using MLflow client
# MAGIC We'll fetch the original R model and then register the python instantiation of that model. Update the widgets above this notebook to the appropriate R model name and version you want to register.

# COMMAND ----------

mlflow_client = MlflowClient()

model_uri = f"models:/{r_model_name}/{r_model_version}"
model_path = ModelsArtifactRepository(model_uri).download_artifacts(artifact_path="")

# COMMAND ----------

mlflow.set_experiment("/Shared/glm-example/glm-python-model")

with mlflow.start_run():
  loaded_model = tf.keras.models.load_model(f"{model_path}/model.h5")
  mlflow.keras.log_model(loaded_model, "glm-models/", extra_pip_requirements=["protobuf<4.0.0"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Python model
# MAGIC We've successfully logged the R model as a python model. Now let's register it appropriately so we may serve this model.

# COMMAND ----------

# MAGIC %md
# MAGIC First, extract the id of the run above, logging our model.

# COMMAND ----------

python_model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
registered_model_version = mlflow.register_model(python_model_uri, python_model_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC If you want a quick sanity check, here's how to load the now registered model:
# MAGIC ```python
# MAGIC python_model_version = registered_model_version.version
# MAGIC 
# MAGIC model = mlflow.pyfunc.load_model(model_uri=f"models:/{python_model_nam}/{python_model_version}")
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the Python model via Model Serving
# MAGIC Next, enable the [model serving](https://docs.databricks.com/applications/mlflow/model-serving.html#model-serving-from-model-registry) for this registered model via the UI. Once the model is served, any subsequent versions added to the registered model will also be exposed as a REST API endpoint.
