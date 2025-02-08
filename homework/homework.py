# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#



# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os


def limpieza(df):
    # Paso 1.
    df = df.copy()
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.drop('ID', axis=1, inplace=True)
    df.dropna(inplace=True)
    df = df[(df["EDUCATION"]!=0) & (df["MARRIAGE"]!=0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x>4 else x) 
    return df
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
def crear_pipeline(categorical_features, k_features=10):
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
        ], 
        remainder='passthrough' 
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', MinMaxScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=k_features)),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ])
    
    return pipeline
def optimizar_hiperparametros(pipeline, x_train, y_train):
    # Definición de la malla de parámetros para la búsqueda
    param_grid = {
        'selector__k': range(1, 11),
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty':['l1','l2'],
        'classifier__solver':['liblinear'],
        "classifier__max_iter": [100,200],
    }

    # Configuración de GridSearchCV para la validación cruzada
    grid_search = GridSearchCV(
        estimator=pipeline,           # Pipeline que incluye el preprocesamiento y el clasificador
        param_grid=param_grid,        # Hiperparámetros a optimizar
        cv=10,                        # 10 divisiones para la validación cruzada
        scoring='balanced_accuracy',  # Métrica: precisión balanceada
        n_jobs=-1,
        refit=True                     # Uso de todos los núcleos disponibles para acelerar el proceso
    )

    
    return grid_search
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
def calcular_metricas(y_true, y_pred, dataset):
    return {
        'type': 'metrics',
        'dataset': dataset,
        'precision': precision_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
def calcular_matriz_confusion(y_true, y_pred, dataset):
    cm = confusion_matrix(y_true, y_pred)
    return {
        'type': 'cm_matrix',
        'dataset': dataset,
        'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

def save_model(path: str, estimator: GridSearchCV):
    with gzip.open(path, 'wb') as f:
        pickle.dump(estimator, f)
def main():
    # Cargar los datasets
    test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")

    # Limpieza
    test_data=limpieza(test_data)
    train_data=limpieza(train_data)
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.    
    # Dividir los datasets
    # Paso 2.
    # Divida los datasets en x_train, y_train, x_test, y_test.
    x_train=train_data.drop('default', axis=1)
    y_train=train_data['default']
    x_test=test_data.drop('default', axis=1)
    y_test=test_data['default']

    # Identificar variables categóricas
    categorical_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Crear y entrenar el pipeline
    pipeline = crear_pipeline(categorical_features)

    # Optimización de hiperparámetros usando validación cruzada
    grid_search = optimizar_hiperparametros(pipeline, x_train, y_train)
    # Ajustar el modelo con la mejor combinación de hiperparámetros
    grid_search.fit(x_train, y_train)

    # Paso 5.
    # Guardar el modelo comprimido con gzip
    path2 = "./files/models/"
    save_model(
        os.path.join(path2, 'model.pkl.gz'),
        grid_search,
    )
   

    pred_train = grid_search.predict(x_train)
    pred_test = grid_search.predict(x_test)

    metrics = [
        calcular_metricas(y_train, pred_train, 'train'),
        calcular_metricas(y_test, pred_test, 'test'),
        calcular_matriz_confusion(y_train, pred_train, 'train'),
        calcular_matriz_confusion(y_test, pred_test, 'test')
    ]
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")
   

if __name__ == "__main__":
    main()