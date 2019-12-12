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
