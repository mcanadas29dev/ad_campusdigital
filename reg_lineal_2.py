import functions as mfunc

df = mfunc.sns.load_dataset('mpg').dropna()
print(df.head())

# queremos predecir mpg (variable objetivo)

# =========================
# 📊 CORRELACIÓN
# =========================
# Calculamos la matriz de correlacion
# Eliminamos las columnas no numericas
df_nuevo = df.drop(columns=['origin','name'])
print(df_nuevo)

# Código que puedes ejecutar (solo referencia)

mfunc.plt.scatter(df['weight'], df['mpg'])
mfunc.plt.xlabel('Peso')
mfunc.plt.ylabel('mpg')
mfunc.plt.show()

corr = df_nuevo.corr()
print(corr) 

# Grafico de mapa de calor 
mfunc.plt.figure(figsize=(8, 6))
mfunc.sns.heatmap(corr, annot=True, cmap="coolwarm")
mfunc.plt.title("Matriz de correlación")
mfunc.plt.show()

# X son las pistas: TV y Sales
X = df[['weight','displacement','horsepower','cylinders']]
# y es lo que intentamos predecir: Sales
y = df['mpg']

# Dividimos el dataset en entrenamiento 80% y test 20%
X_train, X_test, y_train, y_test = mfunc.train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42)
    #stratify=y_encoded)

# Creamos el modelo de regresion lineal
modelo = mfunc.LinearRegression()
# Entrenamos el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)
# Hacemos predicciones con el modelo entrenado
y_pred = modelo.predict(X_test)

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
"""
Con este resultado:
Conclusión: Añadir displacement, horsepower y cylinders no mejoró el modelo. ¿Por qué?  
- Multicolinealidad: Estas variables aportan la misma información que ya daba weight
- En regresión lineal, variables redundantes no ayudan (y pueden incluso empeorar)
Regresión Lineal - R² train: 0.716, R² test: 0.649
Regresión Lineal - RMSE train: 4.23, RMSE test: 4.23
R^2 Score: 0.65
"""