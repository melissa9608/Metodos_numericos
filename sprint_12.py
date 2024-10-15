#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('/datasets/car_data.csv')
print(df.info())

display(df.head())

print(df.columns)

new_names = []
for old_name in df:
    name_lowercase = old_name.lower()
    new_names.append(name_lowercase)

df.columns = new_names
print(df.columns)

columns_new = {
    'datecrawled': 'date_crawled',
    'vehicletype': 'vehicle_type',
    'registrationyear': 'registration_year',
    'gearbox': 'gear_box',
    'registrationmonth': 'registration_month',
    'fueltype': 'fuel_type',
    'notrepaired': 'not_repaired',
    'datecreated': 'date_created',
    'numberofpictures': 'number_of_pictures',
    'postalcode': 'postal_code',
    'lastseen': 'last_seen'
}
df.rename(columns=columns_new, inplace=True)
print(df.columns)

display(df.head())

print('Valores Nulos:')
print(df.isnull().sum())

columns_to_replace = ['vehicle_type', 'gear_box',
                      'model', 'fuel_type', 'not_repaired']
for columns in columns_to_replace:
    df[columns].fillna('unknown', inplace=True)

print('Valores Nulos:')
print(df.isnull().sum())

print(df.duplicated().sum())

df = df.drop_duplicates()
print(df.duplicated().sum())

display(df.sample(10))

print(df.info())

df_sample = df.sample(n=10000, random_state=12345)
df_sample = df_sample.drop(
    columns=['last_seen', 'number_of_pictures', 'postal_code', 'date_crawled'])
df_sample = pd.get_dummies(df_sample, columns=[
                           'vehicle_type', 'gear_box', 'model', 'fuel_type', 'brand', 'not_repaired', 'date_created'], drop_first=True)

features = df_sample.drop(columns=['price'], axis=1)
target = df_sample['price']

features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)

# Lineal Regression
model_lr = LinearRegression()
model_lr.fit(features_train, target_train)
predictions_lr_valid = model_lr.predict(features_valid)
mse_lr = mean_squared_error(target_valid, predictions_lr_valid)
print(f'MSE Regresión Lineal: {mse_lr}')

# Decision Tree
depth_opc = [1, 3, 5, 7, 10]
best_mse = float('inf')
best_model = None
best_params = {}

for depth in depth_opc:
    model_dt = DecisionTreeRegressor(random_state=12345, max_depth=depth)
    model_dt.fit(features_train, target_train)
    predictions_dt_valid = model_dt.predict(features_valid)
    mse_dt = mean_squared_error(target_valid, predictions_dt_valid)
    if mse_dt < best_mse:
        best_mse = mse_dt
        best_model = model_dt
        best_params = {'model': 'DecisionTree', 'max_depth': depth}

print(f'Mejores hiperparámetros: {best_params}')
print(f'Menor MSE en el conjunto de validación: {best_mse}')

# Random Forest
n_estimators_opc = [10, 20, 50, 100]
best_mse = float('inf')
best_model = None
best_params = {}

for n_estimators in n_estimators_opc:
    model_rf = RandomForestRegressor(
        random_state=54321, n_estimators=n_estimators, n_jobs=-1)
    model_rf.fit(features_train, target_train)
    predictions_rf_valid = model_rf.predict(features_valid)
    mse_rf = mean_squared_error(target_valid, predictions_rf_valid)
    if mse_rf < best_mse:
        best_mse = mse_rf
        best_model = model_rf
        best_params = {'model': 'RandomForest', 'n_estimators': n_estimators}

print(f'Mejores hiperparámetros: {best_params}')
print(f'Menor MSE en el conjunto de validación: {best_mse}')

# CatBoost
depth_opc = [1, 3, 5, 7]
estimators_opc = [10, 50, 100]
best_mse = float('inf')
best_model = None
best_params = {}

for depth in depth_opc:
    for n_estimators in estimators_opc:
        model_cb = CatBoostRegressor(
            learning_rate=0.1, depth=depth, iterations=n_estimators, random_state=12345, verbose=0)
        model_cb.fit(features_train, target_train)
        predictions_cb_valid = model_cb.predict(features_valid)
        mse_cb = mean_squared_error(target_valid, predictions_cb_valid)
        if mse_cb < best_mse:
            best_mse = mse_cb
            best_model = model_cb
            best_params = {'model': 'CatBoost',
                           'depth': depth, 'n_estimators': n_estimators}

print(f'Mejores hiperparámetros: {best_params}')
print(f'Menor MSE en el conjunto de validación: {best_mse}')

# Datos NO escalados
# Lineal Regression
start_time = time.time()
model_lr = LinearRegression()
model_lr.fit(features_train, target_train)
train_time_lr = time.time() - start_time

start_time = time.time()
predictions_lr = model_lr.predict(features_valid)
prediction_time_lr = time.time() - start_time

predictions_lr = model_lr.predict(features_valid)
mse_lr = mean_squared_error(target_valid, predictions_lr)
mae_lr = mean_absolute_error(target_valid, predictions_lr)
r2_lr = r2_score(target_valid, predictions_lr)

# Decision Tree
start_time = time.time()
model_dt = DecisionTreeRegressor(random_state=12345, max_depth=5)
model_dt.fit(features_train, target_train)
train_time_dt = time.time() - start_time

start_time = time.time()
predictions_dt = model_dt.predict(features_valid)
prediction_time_dt = time.time() - start_time

predictions_dt = model_dt.predict(features_valid)
mse_dt = mean_squared_error(target_valid, predictions_dt)
mae_dt = mean_absolute_error(target_valid, predictions_dt)
r2_dt = r2_score(target_valid, predictions_dt)

# Random Forest
start_time = time.time()
model_rf = RandomForestRegressor(random_state=12345, n_estimators=100)
model_rf.fit(features_train, target_train)
train_time_rf = time.time() - start_time

start_time = time.time()
predictions_rf = model_rf.predict(features_valid)
prediction_time_rf = time.time() - start_time

predictions_rf = model_rf.predict(features_valid)
mse_rf = mean_squared_error(target_valid, predictions_rf)
mae_rf = mean_absolute_error(target_valid, predictions_rf)
r2_rf = r2_score(target_valid, predictions_rf)

# CatBoost
start_time = time.time()
model_cb = CatBoostRegressor(
    iterations=100, depth=7, learning_rate=0.1, loss_function='RMSE', verbose=0)
model_cb.fit(features_train, target_train)
train_time_cb = time.time() - start_time

start_time = time.time()
predictions_cb = model_cb.predict(features_valid)
prediction_time_cb = time.time() - start_time

predictions_cb = model_cb.predict(features_valid)
mse_cb = mean_squared_error(target_valid, predictions_cb)
mae_cb = mean_absolute_error(target_valid, predictions_cb)
r2_cb = r2_score(target_valid, predictions_cb)

print(f'Regresión Lineal \n- Tiempo de Predicción: {prediction_time_lr:.4f}s\n- Tiempo de Entrenamiento: {
      train_time_lr:.4f}s\n- MSE: {mse_lr:.4f}\n- MAE: {mae_lr:.4f}\n- R2: {r2_lr:.4f}')
print(f'Árbol de Decisión \n- Tiempo de Predicción: {prediction_time_dt:.4f}s\n- Tiempo de Entrenamiento: {
      train_time_dt:.4f}s\n- MSE: {mse_dt:.4f}\n- MAE: {mae_dt:.4f}\n- R2: {r2_dt:.4f}')
print(f'Bosque Aleatorio \n- Tiempo de Predicción: {prediction_time_rf:.4f}s\n- Tiempo de Entrenamiento: {
      train_time_rf:.4f}s\n- MSE: {mse_rf:.4f}\n- MAE: {mae_rf:.4f}\n- R2: {r2_rf:.4f}')
print(f'CatBoost \n- Tiempo de Predicción: {prediction_time_cb:.4f}s\n- Tiempo de Entrenamiento: {
      train_time_cb:.4f}s\n- MSE: {mse_cb:.4f}\n- MAE: {mae_cb:.4f}\n- R2: {r2_cb:.4f}')

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_valid = scaler.transform(features_valid)

# Conclusiones:
# El MSE (Mean Squared Error) mide la diferencia entre los valores predichos y los valores reales elevados al cuadrado.
# Un MSE bajo indica que las predicciones están cerca de los valores reales, mientras que un MSE alto significa que las
# predicciones están más alejadas.

# El MAE (Mean Absolute Error) mide el promedio de las diferencias absolutas entre las predicciones y los valores reales.
# Un MAE bajo indica que las predicciones son más precisas en promedio.

# R2 (R-Squared) mide la proporción de la variación en la variable dependiente que es explicada por el modelo.
# Un valor de R2 cercano a 1 indica un buen ajuste del modelo a los datos.

# Los modelos que tienen valores bajos en MSE y MAE son preferibles, ya que indican que el modelo hace predicciones
# más precisas. Un R2 alto sugiere que el modelo es eficaz para explicar la variabilidad en los datos.

# Comparación de los modelos:
# 1. **CatBoost**:
#   - MSE: 4064543.0934 (menor valor, lo que indica alta precisión)
#   - MAE: 1273.1396 (muy cercano a los mejores)
#   - R2: 0.7997 (el valor más alto, lo que indica mejor capacidad de explicación de la variabilidad)
#   - Tiempo de entrenamiento y predicción: buen balance, sin ser el más rápido, sigue siendo eficiente.
#   - Conclusión: **El mejor modelo** en términos de calidad de predicción y tiempo de entrenamiento.

# 2. **Árbol de Decisión**:
#   - MSE: más alto comparado con otros modelos, por lo tanto, menos preciso.
#   - MAE y R2: peores que CatBoost y Bosque Aleatorio.
#   - Tiempo: el más rápido en entrenamiento y predicción, pero a costa de la precisión.
#   - Conclusión: Útil si se necesita rapidez en predicción, pero se sacrifica precisión.

# 3. **Bosque Aleatorio**:
#   - MSE y MAE: resultados decentes, mejores que el Árbol de Decisión pero peores que CatBoost.
#   - Tiempo: más lento en entrenamiento y predicción.
#   - Conclusión: Una opción equilibrada, pero no tan eficiente como CatBoost.

# 4. **Regresión Lineal**:
#   - MSE y MAE: peores que otros modelos.
#   - Tiempo: más rápido que el Bosque Aleatorio en predicción, pero no tan rápido como el Árbol de Decisión.
#   - Conclusión: No es la mejor opción ni en términos de precisión ni velocidad.
