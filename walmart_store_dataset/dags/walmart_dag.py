from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pipelines.main import train_model, predict_model

dag = DAG('walmart_sales_pipeline',
          start_time=datetime(2026,3,15),
          schedule_interval='@weekly')

train_task = PythonOperator(task_id ='train_model',python_callable=train_model, dag=dag)
predict_task = PythonOperator(task_id='predict_model',python_callable=predict_model, dag=dag)

train_task >> predict_task
