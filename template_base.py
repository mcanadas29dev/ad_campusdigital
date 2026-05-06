"""
TEMPLATE BASE - CAMBIAR SOLO MODEL_TYPE
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, mean_squared_error


# =========================
# 🔥 CAMBIAR SOLO ESTO
# =========================
MODEL_TYPE = "logistic"   # "logistic" | "tree" | "linear"


# =========================
# 📂 CARGA DE DATOS
# =========================
df = pd.read_csv("ejemplos/iris.csv")


# =========================
# ⚙️ PREPROCESADO
# =========================
X = df.drop(columns=["especie"])
y = df["especie"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# =========================
# 🤖 SELECCIÓN DE MODELO
# =========================
if MODEL_TYPE == "logistic":
    model = LogisticRegression(max_iter=200)

elif MODEL_TYPE == "tree":
    model = DecisionTreeClassifier()

elif MODEL_TYPE == "linear":
    model = LinearRegression()

else:
    raise ValueError("Modelo no válido")


# =========================
# 🧠 ENTRENAMIENTO
# =========================
model.fit(X_train, y_train)


# =========================
# 🔮 PREDICCIÓN
# =========================
y_pred = model.predict(X_test)


# =========================
# 📏 EVALUACIÓN
# =========================
if MODEL_TYPE in ["logistic", "tree"]:
    print("Accuracy:", accuracy_score(y_test, y_pred))
else:
    print("MSE:", mean_squared_error(y_test, y_pred))


# =========================
# 🧪 PREDICCIÓN NUEVA
# =========================
ejemplo = [[4.4, 3.0, 1.3, 0.2]]

pred = model.predict(ejemplo)

if MODEL_TYPE in ["logistic", "tree"]:
    pred = le.inverse_transform(pred.astype(int))

print("Predicción:", pred)