from datetime import datetime
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 3),
    # Add other relevant default arguments
}

dag = DAG('spark_dataframe_workflow', default_args=default_args, schedule_interval=None)

# Submitting a Spark job using SparkSubmitOperator
spark_task = SparkSubmitOperator(
    task_id='spark_job_task',
    application='spark_test.py',
    conn_id='spark_default',  # Connection ID to your Spark cluster
    dag=dag,
)



# Set task dependencies
spark_task >> download_dataframe