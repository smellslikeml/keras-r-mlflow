#!/bin/bash

# This bash script is a simple example testing on a condition 
# to determine which REST API endpoint url gets called 
# for inference. It assumes your data is stored in data.json and 
# a PAT with permissions to access the service
# referenced here as DATABRICKS_TOKEN.

echo -n "Enter a number between 1 & 10: "
read VAR

if [[ $VAR -gt 5 ]]
then
  echo "The variable is greater than 5. Call model version 1"

  curl -u token:$DATABRICKS_TOKEN -X POST \
  -H "Content-Type: application/json" \
  -d@data.json \
  https://e2-demo-field-eng.cloud.databricks.com/model/glm-python-model/1/invocations

else
  echo "The variable is equal or less than 5. Call model version 2"

  curl -u token:$DATABRICKS_TOKEN -X POST \
  -H "Content-Type: application/json" \
  -d@data.json \
  https://e2-demo-field-eng.cloud.databricks.com/model/glm-python-model/2/invocations

fi
