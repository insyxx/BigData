# Datos Masivos - Unidad 3

 - **Tecnológico Nacional de México**
 - **Instituto Tecnológico de Tijuana**
 - **Subdirección académica**
 - **Departamento de Sistemas y Computación**
 - **Semestre**: AGOSTO- DICIEMBRE 2019
 - **Ingeniería en Tecnologías de la Información y Comunicaciones**
 - **Materia**: Datos Masivos
 - **Unidad**: 3
 - **Nombre**: Rocha Zepeda Jaime
 - **No. Control**: 16211349
 - **Docente**: Dr. Jose Christian Romero Hernandez
 

 # Practicas
**Practica 1**

**Introducción**
En el siguiente documento veremos una pequeña explicación de lo que es la regresión logística y un pequeño programa en el lenguaje de programación scala.

**Explicación:**
La regresión logística es un tipo de análisis de regresión utilizado para predecir el resultado de una variable categórica (una variable que puede adoptar un número limitado de categorías) en función de las variables independientes o predictoras. Es útil para modelar la probabilidad de un evento ocurriendo como función de otros factores. El análisis de regresión logística se enmarca en el conjunto de Modelos Lineales Generalizados (GLM por sus siglas en inglés) que usa como función de enlace la función logit. Las probabilidades que describen el posible resultado de un único ensayo se modelan, como una función de variables explicativas, utilizando una función logística.

**Codigo**
//Importamos librerias necesarias con las que vamos a trabajar
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.types.DateType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.log4j._
 
//Elimina varios avisos de warnings/errores innecesarios
Logger.getLogger("org").setLevel(Level.ERROR)
 
val spark = SparkSession.builder().getOrCreate()
//Creacion del dataframe para cargar el archivo csv
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
//Imprimimos el esquema del dataframe para visualizarlo
data.printSchema()
//Imprime la primera linea de datos del csv
data.head(1)
data.select("Clicked on Ad").show()
 
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
//Tomamos nuestros datos mas relevantes a una variables y tomamos clicked on ad como nuestra label
//Renombre la columna "Clicked on Ad" a "label"
val logregdataall = timedata.select(data("Clicked on Ad").as("label"),$"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Hour",$"Male")
//Toma la siguientes columnas como features "Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"
val feature_data = data.select($"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Timestamp",$"Male")
 
 
 
 
// se crea una nueva clolumna llamada "Hour" del Timestamp conteniendo la  "Hour"
val logregdataal = (data.withColumn("Hour",hour(data("Timestamp")))
val logregdataal = logregdataall.na.drop()
 
// se crea un nuevo objecto VectorAssembler llamado assembler para los feature
val assembler = new VectorAssembler().setInputCols(Array("Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Hour","Male")).setOutputCol("features")
//Utilizamos randomSplit para crear datos de train y test divididos en 70/30
val Array(training, test) = logregdataall.randomSplit(Array(0.7, 0.3), seed = 12345)
 
// Creamos un nuevo objeto de  LogisticRegression llamado lr
val lr = new LogisticRegression()
// Creamos un nuevo  pipeline con los elementos: assembler, lr
val pipeline = new Pipeline().setStages(Array(assembler,lr))
// Ajuste (fit) el pipeline para el conjunto de training.
val model = pipeline.fit(training)
//resultados de las pruebas con nuestro modelo
val results = model.transform(test)
 
// Convierte los resutalos de prueba (test) en RDD utilizando .as y .rdd
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
 
// Inicializa un objeto MulticlassMetrics 
val metrics = new MulticlassMetrics(predictionAndLabels)
 
//Imprimimos nuestras metricas y la presicion de los calculos
println("Confusion matrix:")
println(metrics.confusionMatrix)
metrics.accuracy

**Practica 2**

**Introducción**
En el siguiente documento hablaremos un poco de el algoritmo k medias y un pequeño programa en el lenguaje de programación scala.

**Explicación:**
K-means es un algoritmo de clasificación no supervisada (clusterización) que agrupa objetos en k grupos basándose en sus características. El agrupamiento se realiza minimizando la suma de distancias entre cada objeto y el centroide de su grupo o cluster. Se suele usar la distancia cuadrática.

**Código**
import org.apache.spark.sql.SparkSession
 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
 
val spark = SparkSession.builder().getOrCreate()
 
import org.apache.spark.ml.clustering.KMeans
 
val dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")
 
//trains the k-means model
val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(dataset)
 
// Evaluate clustering by calculate Within Set Sum of Squared Errors.
val WSSE = model.computeCost(dataset)
println(s"Within set sum of Squared Errors = $WSSE")
 
// Show results
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

# Evaluación 
**Examen 1**

**Introduccion**
En el siguiente documento observaremos la aplicación de un algoritmo de machine learning llamado knn en el cual probaremos con un pequeño dataset.

**Explicación **

KNN es un método de clasificación supervisada (Aprendizaje, estimación basada en un conjunto de entrenamiento y prototipos)
El algoritmo KNN asume que existen cosas similares en la proximidad. En otras palabras, cosas similares están cerca unas de otras, en otras palabras K-NN captura la idea de similitud (a veces llamada distancia, proximidad o cercanía).

Existen muchas fórmulas para calcular la distancia, y una podría ser preferible dependiendo del problema que se esté resolviendo. Sin embargo, la distancia en línea recta (también llamada distancia euclidiana) es una opción popular y familiar.

Objetivo a resolver
El objetivo de este examen es tratar de agrupar los clientes de regiones específicas de un distribuidor al mayoreo. Esto en base a las ventas de algunas categorías de productos.



**Codigo**

// 1. Se importa la sesion en Spark
import org.apache.spark.sql.SparkSession
{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.log4j._
// 2. Se minimizan los errores
Logger.getLogger("org").setLevel(Level.ERROR)
// 3. Creamos la instancia de sesion en Spark
val spark = SparkSession.builder().getOrCreate()
// 4. Se importa la libreria de Kmeans para el algoritmo
import org.apache.spark.ml.clustering.KMeans
// 5. Cargamos el dataset de Wholesale Customers Data
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale-customers-data.csv")
// 6. Seleccionamos las columnas que seran los datos "feature"
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
// 8. Se crea un nuevo objeto para las columnas de caractersiticas
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
// 7. Se importa Vector Assembler para el manejo de datos
import org.apache.spark.ml.feature
// 9. Se utiliza el objetivo "assembler" para transformar "feature_data"
val traning = assembler.transform(feature_data)
// 10. Se crea el modelo con K = 3
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(traning)
// Evaluamos el cluster calculando en los errores cuadraticos
val WSSSE = model.computeCost(traning)
println(s"Within Set Sum of Squared Errors = $WSSSE")
// Resultado
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
