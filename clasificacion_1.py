import functions as mfunc

# carga el dataset

df = mfunc.pd.read_csv('titanic.csv').dropna()
print(df.head())

le = mfunc.LabelEncoder()
# Fit transforma las etiquetas: femenino -> 0, masculino -> 1 (suele ser por orden alfabético)
df['Sex'] = le.fit_transform(df['Sex'])

# Para comenzar, vamos a utilizar estas 3 variables 
# Objetio: Survived (intentar predecir con otras variables)
# Sex: 0 femenino 1 masculino
# Pclass: Clase del pasajero (1,2,3)

# =========================
# 📊 CORRELACIÓN
# =========================
# Calculamos la matriz de correlacion
# Eliminamos las columnas no numericas
df_nuevo = df.drop(columns=['Ticket','Name','PassengerId','Cabin','Embarked'])
print(df_nuevo)

# Código que puedes ejecutar (solo referencia)

corr = df_nuevo.corr()
print(corr) 

# Grafico de mapa de calor 
mfunc.plt.figure(figsize=(8, 6))
mfunc.sns.heatmap(corr, annot=True, cmap="coolwarm")
mfunc.plt.title("Matriz de correlación")
mfunc.plt.show()



X = df_nuevo.drop(columns=["Survived"])
y = df_nuevo["Survived"]

X_train, X_test, y_train, y_test = mfunc.train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = mfunc.LogisticRegression(max_iter=200)
# =========================
# 🧠 ENTRENAMIENTO
# =========================
model.fit(X_train, y_train)


# =========================
# 🔮 PREDICCIÓN
# =========================
y_pred = model.predict(X_test)

print (f"Predicción del Modelo: {y_pred}")


# =========================
#  EVALUACIÓN 
# =========================


# Accuracy en train y test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

acc_train = mfunc.accuracy_score(y_train, y_train_pred)
acc_test = mfunc.accuracy_score(y_test, y_test_pred)

print(f"Accuracy train: {acc_train:.3f} ({acc_train*100:.1f}%)")
print(f"Accuracy test:  {acc_test:.3f} ({acc_test*100:.1f}%)")

# Matriz de confusión en test
cm = mfunc.confusion_matrix(y_test, y_test_pred)
print(f"""
Matriz de confusión (test):
    VP = {cm[1,1]}  (predijo sobrevive y sí sobrevivió)
    VN = {cm[0,0]}  (predijo no sobrevive y no sobrevivió)
    FP = {cm[0,1]}  (predijo sobrevive pero no sobrevivió)
    FN = {cm[1,0]}  (predijo no sobrevive pero sí sobrevivió)
""")

# Coeficientes del modelo (para ver qué variable importa más)
coeficientes = mfunc.pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': model.coef_[0]
})
print(f"Coeficientes:\n{coeficientes.sort_values('Coeficiente', ascending=False)}")


