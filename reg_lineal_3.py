import functions as mfunc

# Cargar el dataset (limpio)

df = mfunc.pd.read_csv('boston.csv') # lo puedo leer si me situo en el directorio ejemplos
print(df.head())
print(df.info())

# Limpiar columnas de texto innecesarias

# visualizar datos, matriz de correlacion, heatmap, pairplot. 
corr = df.corr()
print(corr) 
# Grafico de mapa de calor 
mfunc.plt.figure(figsize=(8, 6))
mfunc.sns.heatmap(corr, annot=True, cmap="coolwarm")
mfunc.plt.title("Matriz de correlación")
mfunc.plt.show()
# Grafico Pirplot

cols = ["medv", "rm", "lstat", "ptratio", "nox"]
mfunc.plt.figure(figsize=(8, 6))
mfunc.sns.pairplot(df[cols])
mfunc.plt.title("Pairplot")
mfunc.plt.show()

"""
Correlaciones detectadas:
- lstat - medv correlacion- -0.737663 
- medv - rm correlacion +  0.695360
"""

# Entrenamiento
# X son las pistas: TV y Sales
X = df[['lstat','rm','zn','age']]
# y es lo que intentamos predecir: Sales
y = df['medv']

# Dividimos el dataset en entrenamiento 80% y test 20%
X_train, X_test, y_train, y_test = mfunc.train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42)

# Evaluacion
# Creamos el modelo de regresion lineal
modelo = mfunc.LinearRegression()
# Entrenamos el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)
# Hacemos predicciones con el modelo entrenado
y_pred = modelo.predict(X_test)
# Evaluamos el modelo utilizando el coeficiente de determinación R^2
# =========================
# 🔍 R² EN TRAIN Y TEST (para detectar overfitting)
# =========================
r2_train = modelo.score(X_train, y_train)
r2_test = modelo.score(X_test, y_test)
print(f"Regresión Lineal - R² train: {r2_train:.3f}, R² test: {r2_test:.3f}")

rmse_train = mfunc.root_mean_squared_error(y_train, modelo.predict(X_train))
rmse_test = mfunc.root_mean_squared_error(y_test, y_pred)
print(f"Regresión Lineal - RMSE train: {rmse_train:.2f}, RMSE test: {rmse_test:.2f}")

# Evaluamos el modelo utilizando el coeficiente de determinación R^2
r2_score = modelo.score(X_test, y_test)
print(f"R^2 Score: {r2_score:.2f}")

# Crear y entrenar Ridge (alpha=1 es valor por defecto)
ridge = mfunc.Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Métricas en entrenamiento y prueba
r2_ridge_train = ridge.score(X_train, y_train)
r2_ridge_test = ridge.score(X_test, y_test)

print(f"Ridge - R² train: {r2_ridge_train:.3f}, R² test: {r2_ridge_test:.3f}")

rmse_ridge_train = mfunc.root_mean_squared_error(y_train, ridge.predict(X_train))
rmse_ridge_test = mfunc.root_mean_squared_error(y_test, ridge.predict(X_test))

print(f"Ridge - RMSE train: {rmse_ridge_train:.2f}, RMSE test: {rmse_ridge_test:.2f}")



# Conclusion
