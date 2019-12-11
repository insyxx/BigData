//Libreria spark
// 1. Se importa la sesion en Spark
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
// 7. Se importa Vector Assembler para el manejo de datos
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
// 4. Se importa la libreria de Kmeans para el algoritmo
import org.apache.spark.ml.clustering.KMeans

import org.apache.spark.ml.linalg.Vectors
import org.apache.log4j._

// 2. Se minimizan los errores
//Para que no marque errores 
Logger.getLogger("org").setLevel(Level.ERROR)

// 3. Creamos la instancia de sesion en Spark
// Cargamos el csv 
val spark = SparkSession.builder().getOrCreate()

// 5. Cargamos el dataset de Wholesale Customers Data
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

// 6. Seleccionamos las columnas que seran los datos "feature"
//Selecionamos las columnas que vamos a entrenar
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")
// 8. Se crea un nuevo objeto para las columnas de caractersiticas
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

// 9. Se utiliza el objetivo "assembler" para transformar "feature_data"
//Entrenamos la data que colocamos en el vectorassembler
val traning = assembler.transform(feature_data)

// 10. Se crea el modelo con K = 3
//La data entrenada la pasamos al modelo 
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(traning)
// Evaluamos el cluster calculando en los errores cuadraticos
val WSSSE = model.computeCost(traning)
println(s"Within Set Sum of Squared Errors = $WSSSE")
//Resultado
println("Cluster Centers: ")
model.clusterCenters.foreach(println)