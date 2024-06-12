import train_model as tm
import pandas as pd
import numpy as np
from keras.models import load_model
from train_model import split, modelA, modelB, modelC, evaluate
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key

symbol = 'AAPL'  # Example: Apple stock symbol

def np_array_to_list(arr):
    if arr.ndim == 1:
        return arr.tolist()
    else:
        return [np_array_to_list(a) for a in arr]

def __call_pull_data(ti):
    data = tm.show_table(symbol).to_parquet('data.parquet')
    ti.xcom_push(key = "df_path", value='data.parquet')
def __call_preprocessing(ti):
    data = ti.xcom_pull(
        task_ids= "pull_data",
        key = 'df_path'
    )
    data = pd.read_parquet(data)
    print(data)
    x, y = tm.preprocessing(data)
    ti.xcom_push(key = "x_path", value=np_array_to_list(x))
    ti.xcom_push(key = "y_path", value= np_array_to_list(y))

def __call_split(ti):
    x = ti.xcom_pull(
        task_ids= "preprocessing",
        key = 'x_path'   
    )
    y = ti.xcom_pull(
        task_ids= "preprocessing",
        key = 'y_path'   
    )
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    x_train, x_test, y_train, y_test = split(x, y)
    ti.xcom_push(key = "x_train_path", value=np_array_to_list(x_train))
    ti.xcom_push(key = "y_train_path", value=np_array_to_list(y_train))

    ti.xcom_push(key = "x_test_path", value=np_array_to_list(x_test))
    ti.xcom_push(key = "y_test_path", value=np_array_to_list(y_test))

def __call_modelA(ti):
    x_train = ti.xcom_pull(
        task_ids = "split",
        key = "x_train_path"
    )
    y_train = ti.xcom_pull(
        task_ids = 'split',
        key = "y_train_path"
    )
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    history = modelA(x_train, y_train)
    history.save("model_a.h5")
    ti.xcom_push(key = "model_a_path", value = "model_a.h5")

def __call_modelB(ti):
    x_train = ti.xcom_pull(
        task_ids = "split",
        key = "x_train_path"
    )
    y_train = ti.xcom_pull(
        task_ids = 'split',
        key = "y_train_path"
    )
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    history = modelB(x_train, y_train)
    history.save("model_b.h5")
    ti.xcom_push(key = "model_b_path", value = "model_b.h5")


def __call_modelC(ti):
    x_train = ti.xcom_pull(
        task_ids = "split",
        key = "x_train_path"
    )
    y_train = ti.xcom_pull(
        task_ids = 'split',
        key = "y_train_path"
    )
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    history = modelC(x_train, y_train)
    history.save("model_c.h5")
    ti.xcom_push(key = "model_c_path", value = "model_c.h5")

def __call_choose_best_model(ti):
    model_a = ti.xcom_pull(
        task_ids = "training_model_A",
        key = "model_a_path"
    )
    model_b = ti.xcom_pull(
        task_ids = "training_model_B",
        key = "model_b_path"
    )
    model_c = ti.xcom_pull(
        task_ids = "training_model_C",
        key = "model_c_path"
    )
    models = [load_model(model_a), load_model(model_b), load_model(model_c)]
    
    x_test = ti.xcom_pull(
        task_ids = "split",
        key = "x_test_path"
    )
    y_test = ti.xcom_pull(
        task_ids = 'split',
        key = "y_test_path"
    )
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return tm.choose_best_model(models, x_test, y_test) + 1



with DAG("stock_dag",
         start_date=datetime(2023, 12, 22),
         schedule_interval="@once",
         catchup=False,
         ) as dag:
    pull_data = PythonOperator(
        task_id="pull_data",
        python_callable=__call_pull_data
    )

    preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable= __call_preprocessing
    )

    split_data = PythonOperator(
        task_id="split",
        python_callable= __call_split
    )

    training_model_A = PythonOperator(
        task_id="training_model_A",
        python_callable= __call_modelA
    )

    training_model_B = PythonOperator(
        task_id="training_model_B",
        python_callable= __call_modelB
    )

    training_model_C = PythonOperator(
        task_id="training_model_C",
        python_callable= __call_modelC
    )

    choose_best_model = PythonOperator(
        task_id="choose_best_model",
        python_callable= __call_choose_best_model
    )

pull_data >> preprocessing >> split_data >> [
    training_model_A, training_model_B, training_model_C] >> choose_best_model
