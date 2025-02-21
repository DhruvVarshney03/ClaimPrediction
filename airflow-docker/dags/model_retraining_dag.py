from airflow import DAG # type: ignore
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
import os
import shutil
import time
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import pandas as pd
import logging
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Define constants
AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow/")
BEST_MODEL_PATH = os.path.join(AIRFLOW_HOME, 'Fast_Furious_Insured/api/models/best_model.keras')
MODEL_DIR = os.path.join(AIRFLOW_HOME, 'Fast_Furious_Insured/api/models/')
TRAIN_DATA_PATH = os.path.join(AIRFLOW_HOME, 'Fast_Furious_Insured/api/processed_data/final_train_data.pkl')

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_data():
    logging.info("Loading data for model training...")
    data = pd.read_pickle(TRAIN_DATA_PATH)
    structured_data = data.iloc[:, :18]
    structured_data = structured_data.drop(columns=[data.columns[0], data.columns[5], data.columns[6]])
    structured_data_scaled = structured_data.values

    image_features = data.iloc[:, 18:].values
    condition_labels = data['Condition'].values
    amount_labels = data['Amount'].values.reshape(-1, 1)

    return image_features, structured_data_scaled, condition_labels, amount_labels

def build_model(image_features, structured_data_scaled):
    logging.info("Building model...")
    image_input = Input(shape=(image_features.shape[1],), name='image_input')
    x = Dense(512, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00095))(image_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dropout(0.2)(x)

    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00095))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dropout(0.2)(x)

    structured_input = Input(shape=(structured_data_scaled.shape[1],), name='structured_input')
    y = Dense(128, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00095))(structured_input)
    y = BatchNormalization()(y)
    y = LeakyReLU(negative_slope=0.1)(y)
    y = Dropout(0.2)(y)

    y = Dense(64, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00095))(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(negative_slope=0.1)(y)
    y = Dropout(0.2)(y)

    combined = Concatenate()([x, y])

    z = Dense(128, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00095))(combined)
    z = BatchNormalization()(z)
    z = LeakyReLU(negative_slope=0.1)(z)
    z = Dropout(0.2)(z)

    z = Dense(64, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.00095))(z)
    z = BatchNormalization()(z)
    z = LeakyReLU(negative_slope=0.1)(z)
    z = Dropout(0.2)(z)

    condition_output = Dense(1, activation='sigmoid', name='condition_output')(z)
    amount_output = Dense(1, activation='linear', name='amount_output')(z)

    model = Model(inputs=[image_input, structured_input], outputs=[condition_output, amount_output])

    model.compile(optimizer=Adam(learning_rate=0.00095),
                  loss={'condition_output': 'binary_crossentropy', 'amount_output': tf.keras.losses.Huber()},
                  metrics={'condition_output': 'accuracy', 'amount_output': 'mse'})

    return model

def rollback_model():
    logging.info("Performing rollback...")
    backup_model_path = os.path.join(MODEL_DIR, f'backup_best_model_{int(time.time())}.keras')
    if os.path.exists(BEST_MODEL_PATH):
        shutil.copy(BEST_MODEL_PATH, backup_model_path)
        logging.info(f"Backup created: {backup_model_path}")

def train_and_log_model():
    logging.info("Starting model training...")
    image_features, structured_data_scaled, condition_labels, amount_labels = load_data()
    model = build_model(image_features, structured_data_scaled)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    with mlflow.start_run():
        mlflow.log_param('learning_rate', 0.00095)
        mlflow.log_param('epochs', 200)
        mlflow.log_param('batch_size', 16)

        history = model.fit(
            [image_features, structured_data_scaled],
            {'condition_output': condition_labels, 'amount_output': amount_labels},
            epochs=200,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )

        mlflow.log_metric('condition_output_accuracy', history.history['condition_output_accuracy'][-1])
        mlflow.log_metric('amount_output_mse', history.history['amount_output_mse'][-1])
        model.save(os.path.join(MODEL_DIR, f'model_{int(time.time())}.keras'))
        mlflow.tensorflow.log_model(model, artifact_path="model")
        logging.info("Model training complete.")

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 2, 18),
}

with DAG('model_retraining_dag', default_args=default_args, schedule_interval='@daily', catchup=False) as dag:
    start_task = DummyOperator(task_id='start')
    train_task = PythonOperator(task_id='train_model', python_callable=train_and_log_model)
    rollback_task = PythonOperator(task_id='rollback_model', python_callable=rollback_model, trigger_rule="one_failed")
    end_task = DummyOperator(task_id='end')
    start_task >> train_task >> rollback_task >> end_task
