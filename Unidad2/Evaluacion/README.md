# Evaluación 
** Examen Unidad 2 **

**Introduccion**


import org.apache.spark.ml.classification.MultilayerPerceptronClassifier 
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
import org.apache.spark.sql.types._ 
import org.apache.spark.ml.Pipeline 
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer

//Se crea la estructura de los datos por columnas
val estructuraIris = StructType(StructField("Datos0", DoubleType, true) ::StructField("Datos1", DoubleType, true) ::StructField("Datos2", DoubleType, true) ::StructField("Datos3",DoubleType, true) ::StructField("Datos4", StringType, true) :: Nil)

//Se carga Iris
val datosIris = spark.read.option("header", "true").schema(estructuraIris)csv("iris.csv")

//Creamos la columna label
val etiqueta = new StringIndexer().setInputCol("Datos4").setOutputCol("label")
//Se crea un arreglo de los datos de las columnas Datos0-3 en la columna features
val ensamblador = new VectorAssembler().setInputCols(Array("Datos0", "Datos1", "Datos2", "Datos3")).setOutputCol("features")

//Se separa los datos en dos grupos, uno para entrenar y el otro para prueba desde nuestro df
//Datos de entrenamiento:70 y prueba:30
val splits = datosIris.randomSplit(Array(0.85, 0.15), seed = 1234L)
val entrenar = splits(0) //entrenar
val prueba = splits(1)  //prueba

//Especificamos las capas de nuestra red neuronal
//Capa entrada:4 neuronas. Capa intermedia:4,5 neuronas. Capa salida:3 neuronas
//Son 4 y 3 salidas: setosa, versicolor, virginica
val capasN = Array[Int](4, 8, 7, 3)

//Se hace la creacion del entrenador y se le dan los parametros
//.setLayers es para cargar las capas de nuestra red neuronal
//.setMaxIter es para indicar el numero maximo de iteraciones
val entrenador = new MultilayerPerceptronClassifier().setLayers(capasN).setBlockSize(128).setSeed(1234L).setMaxIter(100)

//Se genera un pipeline con los datos que necesitamos de la label y las features 
val pipeline = new Pipeline().setStages(Array(etiqueta,ensamblador,entrenador))

//Entrenando el modelo 
val modelo = pipeline.fit(entrenar)

//Calculamos la exactitud en el conjunto prueba
val resultados = modelo.transform(prueba)

//Mostramos el resultado
resultados.show()

//prediciendo la exactitud con un evaluador
val predictionAndLabels = resultados.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

//Imprimimos los resultados de exactitud utilizando un evaluador multiclase
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")



//D).Explique detalladamente la funcion matematica de entrenamiento que utilizo con sus propias palabras
/*
La funcion de entrenamiento es el conjunto de datos  que se divide en una parte utilizada para entrenar el modelo (60%)
y otra parte para las prueba (40%).
Esta funcion mediante un array hacemos las pruebas de entrenamiento mediante un ramdom
asi de esta manera se entrena el algoritmo y se costruye el modelo
*/

//E).Explique la funcion de error que utilizo para el resultado final
/*
 Esta funcion nos sirve para calcular el error de prueba nos ayuda a medir la precisión del modelo utilizando el evaluador.
 y asi imprimir con exactitud el error de nuestro problema.
*/

