import functions as mfunc

TITULO = "Comenzamos regresion lineal"

print("*" * len(TITULO))
print(TITULO)
print("*" * len(TITULO))


datos = mfunc.sns.load_dataset('mpg').dropna()

df = mfunc.pd.DataFrame(datos)
print(df.info())
print(df.head())

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

# Cilindros - desplazamiento están muy correlacionados+  0.95
# Peso - desplazamiento muy correlacionados+ 0.93
# CV potencia - desplazamiento está muy correlacionados+ 0.90
# CV potencia - peso están muy correlacionados+ 0.86 

# Voy a entrenar el modelo con 1 variable weight (peso) para intentar
# predecir si a mas peso + mpg (miles per gallon)

# Intentamos predecir 

# X son las pistas: TV y Sales
X = df[['weight']]
# y es lo que intentamos predecir: Sales
y = df['mpg']

# Entrenamiento del modelo (utilizamos regresion lineal por ser importes)
# le = mfunc.LabelEncoder()
# y_encoded = le.fit_transform(y)

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
# Evaluamos el modelo utilizando el coeficiente de determinación R^2
r2_score = modelo.score(X_test, y_test)
print(f"R^2 Score: {r2_score:.2f}")

# Segundo modelo Arbol de decisión
modelo2 = mfunc.DecisionTreeRegressor(random_state=42)
modelo2.fit(X_train, y_train)
r2_arbol = modelo2.score(X_test, y_test)
print(f"R² Árbol: {r2_arbol:.2f}")

# También compara RMSE
from sklearn.metrics import root_mean_squared_error

rmse_lr = root_mean_squared_error(y_test, y_pred)
rmse_arbol = root_mean_squared_error(y_test, modelo2.predict(X_test))
print(f"RMSE Regresión Lineal: {rmse_lr:.2f}")
print(f"RMSE Árbol: {rmse_arbol:.2f}")

"""
=== REGRESIÓN (mpg) ===
Modelo 1 (Regresión Lineal): R² = 0.65, RMSE = 4.21 <--
Modelo 2 (Árbol de Decisión): R² = 0.35, RMSE = 5.78
El error RMSE es menor en Regresión Lineal (4.21 < 5.78), por lo tanto es más preciso.

Ganador
Regresión Lineal
- R^2 mas alto -> explica mejor los datos 
- RMSE mas bajo -> comete menos error en predicciones
- la relación peso-consumo es lineal (a mas peso, menos mpg)

"""