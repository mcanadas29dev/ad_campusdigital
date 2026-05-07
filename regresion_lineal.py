import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Vamos a ir paso a paso con el fichero Advertising.csv

# Paso 1 - Cargamos el dataset 
# TV => Inversión en publicidad en televisión (en miles de dólares).
# Radio => Inversión en publicidad en radio (en miles de dólares).
# Periodicos escritos => nversión en publicidad en prensa escrita (en miles de dólares).

# =========================
# 📂 CARGA DE DATOS
# =========================
# Sales = Ventas del producto(miles de unidades)
df = pd.read_csv("ejemplos/Advertising.csv")
print(df.head())
print(df.info())

# =========================
# 📊 CORRELACIÓN
# =========================
# Calculamos la matriz de correlacion

matriz = df.corr()
print(matriz) 
# Sales - TV tienen un coef. de correlación de 0.78 positiva, 
# Sales - Radio tienen un coef. de correlación de 0.57 positivo. 
# La TV (input), targe (sales) vamos a intentar predecir que a mayor inversion en TV mayor sales
# inversión en TV mas aumentan las Sales (vemtas)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz, annot=True, cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()


# Intentamos predecir 

# X son las pistas: TV y Sales
X = df[['TV']]
# y es lo que intentamos predecir: Sales
y = df['sales']

# Entrenamiento del modelo (utilizamos regresion lineal por ser importes)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividimos el dataset en entrenamiento 80% y test 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42)
    #stratify=y_encoded)

# Creamos el modelo de regresion lineal
modelo = LinearRegression()
# Entrenamos el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)
# Hacemos predicciones con el modelo entrenado
y_pred = modelo.predict(X_test)
# Evaluamos el modelo utilizando el coeficiente de determinación R^2
r2_score = modelo.score(X_test, y_test)
print(f"R^2 Score: {r2_score:.2f}")


