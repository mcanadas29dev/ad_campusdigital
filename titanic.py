"""
IRIS MODEL COMPLETO (EDA + ML)
"""

import functions as mfunc

# =========================
# 🔥 CAMBIAR SOLO ESTO
# =========================
MODEL_TYPE = "tree"   # "logistic" | "tree" | "linear"


# =========================
# 📂 CARGA DE DATOS
# =========================
df = mfunc.pd.read_csv("ejemplos/Titanic-Dataset.csv")
df_original = df.copy()

print(df.head())
print(df.info())

df = df.drop(columns=['Ticket','Name','PassengerId','Cabin','Embarked'])
print(df)

# Rellenar nulos en edad, voy a poner la moda de las mujeres 
# y la media de hombres a hombres
print(df.info()) # 891 registros y 714 sin edad 

df['Age'] = df.groupby('Sex')['Age'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.median())
)

# convertimos Sex (male=0, female=1)
le = mfunc.LabelEncoder()
# Fit transforma las etiquetas: femenino -> 0, masculino -> 1 (suele ser por orden alfabético)
df['Sex'] = le.fit_transform(df['Sex'])

print(df.info())

## EMPEZAMOS EL EDA DESDE AQUI

# Variable objetivo Survived

corr = df.corr()
print(corr)


# variables correlacionadas parch, Pclass, Sex
mfunc.plt.figure(figsize=(8, 6))
mfunc.sns.heatmap(corr, annot=True, cmap="coolwarm")
mfunc.plt.title("Matriz de correlación")
mfunc.plt.show()

X = df.drop(columns=["Survived"])
y = df["Survived"]

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


"""
 ¿Funciona bien el modelo? ¡Pues regular!
El modelo acierta aproximadamente un 79% de las veces, tanto en entrenamiento como en test. 
Esto quiere decir que de cada 10 personas, acierta 8 y falla 2.
¿Es bueno? Pues para ser el primer modelo y sin haberlo afinado mucho, no está mal. Pero vamos, 
que si fuera un examen, sacaría un 7,9 jajaja. No es un sobresaliente pero aprueba con buena nota.

Lo mejor de todo es que NO hay sobreajuste (overfitting), porque el porcentaje en train y test 
es casi igual (79,1% vs 79,3%). Eso significa que el modelo se comporta igual con datos nuevos 
que con los que usamos para entrenarlo. ¡Bien ahí!
"""

"""
🔍 ¿QUÉ NOS DICE LA MATRIZ DE CONFUSIÓN?
La matriz me ha quedado así:

Predijo que NO sobrevive	Predijo que SÍ sobrevive
En realidad NO sobrevivió	96  (bien)	14  (error)
En realidad SÍ sobrevivió	23  (error)	46  (bien)
Traducción al cristiano:

Aciertos totales: 96 + 46 = 142 personas bien clasificadas

Errores totales: 14 + 23 = 37 personas mal clasificadas

El problema gordo: El modelo se equivoca más prediciendo que alguien NO va a sobrevivir cuando en realidad 
SÍ sobrevivió (23 personas). Esto sería un problema serio si esto fuera un rescate de verdad, porque estaríamos 
dejando atrás a gente que se podía salvar. Prefiere quedarse corto antes que arriesgarse a decir que alguien vive 
y luego no.
"""

"""
¿QUÉ VARIABLES IMPORTAN MÁS? (MIRAD LOS COEFICIENTES)
Aquí viene lo más interesante. Los coeficientes me dicen qué influye más en la supervivencia:

Variable	Coeficiente	Influencia en supervivencia
Sex (Mujer)	    -2.63	 MUY IMPORTANTE (negativo = las mujeres sobreviven más porque el modelo asigna 1 a hombre y 0 a mujer)
Pclass	        -1.03	 Importante (los de primera clase sobreviven más)
SibSp	        -0.25	 Poco importante (tener hermanos/cónyuge baja un poco la supervivencia)
Age	            -0.03	 Muy poca influencia
Fare	        +0.003	 Casi nada (pero positivo: pagar más ayuda mínimamente)

"""
"""
MIS CONCLUSIONES FINALES 

Lo más importante: SER MUJER. El coeficiente del sexo es brutal. Las mujeres tenían muchas más papeletas para salvarse. 
Ya se sabía por la historia ("mujeres y niños primero"), pero aquí se ve clarísimo.

La clase social importa UN MONTÓN. Los de primera clase la tenían mucho más fácil. Tener un coeficiente de -1.03 
en Pclass significa que por cada clase que bajas (1→2→3), empeoras mucho tus probabilidades. 
Ser pobre en el Titanic era una putada.

Viajar solo o acompañado... El modelo dice que tener hermanos/cónyuge (SibSp) o padres/hijos (Parch) te perjudica ligeramente. 
Quizá porque la gente que viajaba sola era más joven y ágil, o porque las familias tenían más dudas y perdían tiempo.

La edad y el precio del billete casi no importan. Me esperaba que los niños tuvieran más probabilidades 
(por lo de "niños primero"), pero el modelo apenas le da importancia a la edad. 
Igual porque los niños iban con sus familias y eso ya se refleja en otras variables.

"""

"""
QUÉ HARÍA PARA MEJORARLO
Si tuviera que seguir currándomelo (esto es un ciclo de mejora continua):
Crear una variable "familia" combinando SibSp + Parch + 1 (contando al propio pasajero). 
Seguro que eso predice mejor que las dos separadas.

Agrupar edades por rangos (niño, adulto, anciano) en lugar de usar la edad como número. 
Porque la relación supervivencia-edad no es lineal.

Añadir el título de la persona (Mr, Mrs, Miss, Master) a partir del nombre. 
Porque no es lo mismo una "Miss" que una "Mrs", eso ya dice algo.

"""

"""El modelo es decentillo pero mejorable. Acierta el 79%, que no está mal para empezar, y lo más importante: 
confirma lo que ya sabíamos de la historia del Titanic: salvaban primero a mujeres y a los de primera clase. 
El resto (edad, familia, precio) influye pero mucho menos.

Si hubiera que quedarse con una idea: el Titanic era muy clasista y machirulo en los protocolos de salvamento. 
Los datos no mienten.

P.D.: Para el trabajo final, igual habría que probar una Random Forest o un XGBoost a ver si mejora el accuracy. 

"""