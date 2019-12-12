# Datos Masivos - Unidad 2

 - **Tecnológico Nacional de México**
 - **Instituto Tecnológico de Tijuana**
 - **Subdirección académica**
 - **Departamento de Sistemas y Computación**
 - **Semestre**: AGOSTO- DICIEMBRE 2019
 - **Ingeniería en Tecnologías de la Información y Comunicaciones**
 - **Materia**: Datos Masivos
 - **Unidad**: 2
 - **Nombre**: Rocha Zepeda Jaime
 - **No. Control**: 16211349
 - **Docente**: Dr. Jose Christian Romero Hernandez
 
  # Practicas
**Practica 1**

**Introducción**
En el siguiente documento hablaremos de los temas correlación, test de hipótesis y resumen

**Explicación de los temas**

Correlación:
El Coeficiente de Correlación de Pearson es una medida de la correspondencia o relación lineal entre dos variables cuantitativas aleatorias. En palabras más simples se puede definir como un índice utilizado para medir el grado de relación que tienen dos variables, ambas cuantitativas.

Test de hipótesis:
Una hipótesis estadística es una asunción relativa a una o varias poblaciones, que puede ser cierta o no. Las hipótesis estadísticas se pueden contrastar con la información extraída de las muestras y tanto si se aceptan como si se rechazan se puede cometer un error.
La hipótesis formulada con intención de rechazarla se llama hipótesis nula y se representa por H0. Rechazar H0 implica aceptar una hipótesis alternativa (H1).

Sumatoria (Resumen)
Proporciona estadística en resúmenes de vectores en el uso de grandes volúmenes de datos, las métricas son:
Máximo
Mínimo
Promedio
Varianza
Número de no ceros
Así como resultados totales

**Codigo correlacion**


// Se declaran las librerias necesarias
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

// Se declaran en la variable "df" los vectores donde tendran los datos a procesar
val data = Seq(
 Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
 Vectors.dense(4.0, 5.0, 0.0, 3.0),
 Vectors.dense(6.0, 7.0, 0.0, 8.0),
 Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
)
// Se tranforma la variable "df" a dataframe donde los datos seran agregados a la columna "features"
val df = data.map(Tuple1.apply).toDF("features")
// Se declara la funcion de correlacion y se empieza a trabajar con ellos
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
println(s"Pearson correlation matrix:\n $coeff1")
// Se manda a imprimir la correlacion
val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
println(s"Spearman correlation matrix:\n $coeff2")

**Codigo Hypothesis testing*

// Se declaran las librerias necesarias
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest

// Se declaran en la variable "data" los vectores donde tendrán los datos a procesar

val data = Seq(
 (0.0, Vectors.dense(0.5, 10.0)),
 (0.0, Vectors.dense(1.5, 20.0)),
 (1.0, Vectors.dense(1.5, 30.0)),
 (0.0, Vectors.dense(3.5, 30.0)),
 (0.0, Vectors.dense(3.5, 40.0)),
 (1.0, Vectors.dense(3.5, 40.0))
)
// Se tranforma la variable "data" a dataframe ("df"), este mismo tendra dos columnas llamadas "label", "features"
val df = data.toDF("label", "features")
// Se declara el modelo y sus se agregan sus respectivos parametros
val chi = ChiSquareTest.test(df, "features", "label").head
println(s"pValues = ${chi.getAs[Vector](0)}")
println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
println(s"statistics ${chi.getAs[Vector](2)}")


**Codigo Summarizer**

// Se importan las librerias necesarias
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer

// Se crear una variable llamada "data" la cual tendra los datos a procesar
val data = Seq(
   (Vectors.dense(2.0, 3.0, 5.0), 1.0),
   (Vectors.dense(4.0, 6.0, 7.0), 2.0)
)
// Se transforma la variable "data" en DataFrame con las columnas "features" , "weight"
val df = data.toDF("features", "weight")
df.show()

//Creamos una sumatoria utilizando los datos del DataFrame
val (meanVal, varianceVal) = df.select(Summarizer.metrics("mean", "variance").summary($"features", $"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()
// Se imprime el resultado
println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")

//Creamos una sumatoria
val (meanVal2, varianceVal2) = df.select(Summarizer.mean($"features"),Summarizer.variance($"features")).as[(Vector, Vector)].first()
// Se imprime el resultado
println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")

**Practica 2**

**Introducción**
En el siguiente documento explicaremos un poco de lo que es árboles de decisión y una pequeña aplicación en el lenguaje de programación scala

**Explicación:**
Un árbol de decisión es un mapa de los posibles resultados de una serie de decisiones relacionadas. Permite que un individuo o una organización comparen posibles acciones entre sí según sus costos, probabilidades y beneficios. Se pueden usar para dirigir un intercambio de ideas informal o trazar un algoritmo que anticipe matemáticamente la mejor opción.
Un árbol de decisión, por lo general, comienza con un único nodo y luego se ramifica en resultados posibles. Cada uno de esos resultados crea nodos adicionales, que se ramifican en otras posibilidades. Esto le da una forma similar a la de un árbol.
Hay tres tipos diferentes de nodos: nodos de probabilidad, nodos de decisión y nodos terminales. Un nodo de probabilidad, representado con un círculo, muestra las probabilidades de ciertos resultados. Un nodo de decisión, representado con un cuadrado, muestra una decisión que se tomará, y un nodo terminal muestra el resultado definitivo de una ruta de decisión.

**Código**
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Se instancia los DataFrame en la variable "data" en el formato "libsvm"
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Se agrega otra columna de indices, donde se tomaron los datos de la columna "label" y se transformaron a datos numericos
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data) // features with > 4 distinct values are treated as continuous.

// Se declararan 2 arreglos, uno tendra los datos de entrenamiento y el otro tendra
// los datos de prueba
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Se declara el Clasificador de árbol de decisión y se le agrega la columna que sera las etiquetas (indices) y indice (caracteristicas)
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

//Convierte las etiquetas indexadas a las originales
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels

//Crea el DT pipeline Agregando los index, label y el arbol juntos
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Se entrena el modelo con los datos del arreglo "trainingData" que es el 70% de los datos totales
val model = pipeline.fit(trainingData)

// Se hacen las predicciones al tomar los datos sobrantes que se llevo "testData" que es el 30%
val predictions = model.transform(testData)

// Se manda a imprimir la etiqueta, sus respectivos valores y la prediccion de la etiqueta
predictions.select("predictedLabel", "label", "features").show(5)

// Evalua el modelo y retorna la  métrica escalar
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
// La variable "accuracy" tomara la acertación que hubo respecto a "predictedLabel" y "label"
val accuracy = evaluator.evaluate(predictions)
// Se manda a imprimir el resultado de error con respecto a la exactitud
println(s"Test Error = ${(1.0 - accuracy)}")

// Se guarda en la variable
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

**Practica 3**

**Introducción**
En el siguiente documento explicaremos un poco de el random forest y una pequeña aplicación en el lenguaje de programación scala.

**Explicación:**
Los Bosques Aleatorios es un algoritmo de Machine Learning flexible y fácil de usar que produce, incluso sin ajuste de parámetros, un gran resultado la mayor parte del tiempo. También es uno de los algoritmos más utilizados, debido a su simplicidad y al hecho de que se puede usar tanto para tareas de clasificación como de regresión.
Los Bosques Aleatorios es un algoritmo de aprendizaje supervisado que, como ya se puede ver en su nombre, crea un bosque y lo hace de alguna manera aleatorio. Para decirlo en palabras simples: el Bosque Aleatorio crea múltiples árboles de decisión y los combina para obtener una predicción más precisa y estable. En general, mientras más árboles en el bosque se vea, más robusto es el bosque.
En este algoritmo se agrega aleatoriedad adicional al modelo, mientras crece los árboles, en lugar de buscar la característica más importante al dividir un nodo, busca la mejor característica entre un subconjunto aleatorio de características. Esto da como resultado una amplia diversidad que generalmente resulta en un mejor modelo.

**Codigo**

Código
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Cargue y analice el archivo de datos, convirtiéndolo en un DataFrame.
val data = spark.read.format("libsvm").load("./sample_libsvm_data.txt")data.show()
// Índice de etiquetas, agregando metadatos a la columna de etiquetas.
// Se ajusta a todo el conjunto de datos para incluir todas las etiquetas en el índice.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// Identifica automáticamente las características categóricas y las indexa.
// Se establecen el maxCategories para que las entidades con > 4 valores distintos se traten como continuas.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Divide los datos en conjuntos de entrenamiento y prueba (30% para pruebas).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Entrena un modelo RandomForest.
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

// Convierte las etiquetas indexadas de nuevo a etiquetas originales.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Indicadores de cadena y bosque en una tubería.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Modelo de tren. Esto también ejecuta los indexadores.
val model = pipeline.fit(trainingData)

// Hacer predicciones.
val predictions = model.transform(testData)

// Seleccione filas de ejemplo para mostrar.
predictions.select("predictedLabel", "label", "features").show(5)

// Seleccione (predicción, etiqueta verdadera) y calcule el error de prueba.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n ${rfModel.toDebugString}")

**Practica 4**

**Introducción**
En el siguiente documento explicaremos un poco de lo que es el Gradient-boosted tree classifier y un pequeña aplicación en el lenguaje de programación scala.

**Explicación:**
Los clasificadores de aumento de gradiente son un grupo de algoritmos de aprendizaje automático que combinan muchos modelos de aprendizaje débiles para crear un modelo predictivo sólido. Los árboles de decisión generalmente se usan al aumentar el gradiente.



**Código**
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Se carga los datos en la variable "data" en formato "libsvm"
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Se agrega una nueva columna "IndexLabel" que tendra todos los datos de la columna "label" y tambien los datos los transforma en datos numericos

val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
// Se agrega una nueva columna "indexedFeatures" que tendra todos los datos de la columna "features" y tambien en datos numericos

val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Se declaran dos arreglos; "trainingData" y "testData" de los cuales tendran 70% y 30% de los datos
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Se declara el modelo y se agregan como parametros "indexedLabel" y "indexedFeatures"
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

// Se convierten las "indexedLabel" a las etiquetas originales
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Se declara el objeto "pipeline" en donde nos ayudara a pasar el codigo por estados, estos mismos estan declarados despues de "Array"
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Se entrena el modelo con los datos de entrenamiento
val model = pipeline.fit(trainingData)

// Se hacen las predicciones con el modelos ya entrenado y con los datos de prueba que representan el 30%
val predictions = model.transform(testData)

// Se mandan a imprimir o se seleccionan algunas columnas y se muestran solo las primerias 5
predictions.select("predictedLabel", "label", "features").show(5)

// Se evalua la precision y se agrega a una variable "accuracy"
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
// Se manda a imprimir el error de precision del modelo
println(s"Test Error = ${1.0 - accuracy}")
// Se manda a imprimir el arbol por medio de condicionales "if and else"
val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")


**Practica 5**

**Introducción**

En el siguiente documento hablaremos un poco de Multilayer perceptron classifier y una pequeña aplicacion en el lenguaje de programación scala.

**Explicación:**
Perceptrón Multicapa
El perceptrón multicapa evoluciona el perceptrón simple y para ello incorpora capas de neuronas ocultas, con esto consigue representar funciones no lineales.
El perceptrón multicapa está compuesto por por una capa de entrada, una capa de salida y n capas ocultas entremedias.
Se caracteriza por tener salidas disjuntas pero relacionadas entre sí, de tal manera que la salida de una neurona es la entrada de la siguiente.
En el perceptrón multicapa se pueden diferenciar una 2 fases:
Propagación en la que se calcula el resultado de salida de la red desde los valores de entrada hacia delante.
Aprendizaje en la que los errores obtenidos a la salida del perceptrón se van propagando hacia atrás (backpropagation) con el objetivo de modificar los pesos de las conexiones para que el valor estimado de la red se asemeja cada vez más al real, este aproximación se realiza mediante la función gradiente del error.

**Codigo**
// Se importan todas la librerias necesarias
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// Se cargan los datos en la variable "data" en un formato "libsvm"
val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")

// Se declara un variable llamada "splits" donde se hacen los cortes de forma aleatoria de los datos de la variable "data"
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
// Se declara la variable "train" donde con ayuda de "splits" tendra el primer parametro que es el 60% de los datos cortados
val train = splits(0)
// Se declara la variable "test" donde con ayuda de "splits" tendra el primer parametro que es el 40% de los datos cortados
val test = splits(1)

// Se especifican las capas de la red neuronal
// Capa de entrada de tamaño 4 (características), dos intermedios de tamaño 5 y 4 y salida de tamaño 3 (clases)
val layers = Array[Int](4, 5, 4, 3)

// Se declara el modelo y se agregan los parametros necesarios para su funcionamiento
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// Se entrena el modelo con los datos de entrenamiento
val model = trainer.fit(train)

// Se evalua y despliega resultados
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

// Se imprime el error de la precision 
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

**Practica 6**

**Introducción**

En el siguiente documento explicaremos un poco de lo que es Linear Support Vector Machine y una pequeña aplicación en el lenguaje de programación scala.

**Explicación:**
Linear Support Vector Machine
Una máquina de vectores de soporte (SVM) es un algoritmo de aprendizaje supervisado que se puede emplear para clasificación binaria o regresión. Las máquinas de vectores de soporte son muy populares en aplicaciones como el procesamiento del lenguaje natural, el habla, el reconocimiento de imágenes y la visión artificial.
Una máquina de vectores de soporte construye un hiperplano óptimo en forma de superficie de decisión, de modo que el margen de separación entre las dos clases en los datos se amplía al máximo. Los vectores de soporte hacen referencia a un pequeño subconjunto de las observaciones de entrenamiento que se utilizan como soporte para la ubicación óptima de la superficie de decisión.
Las máquinas de vectores de soporte pertenecen a una clase de algoritmos de Machine Learning denominados métodos kernel y también se conocen como máquinas kernel.
El entrenamiento de una máquina de vectores de soporte consta de dos fases:
Transformar los predictores (datos de entrada) en un espacio de características altamente dimensional. En esta fase es suficiente con especificar el kernel; los datos nunca se transforman explícitamente al espacio de características. Este proceso se conoce comúnmente como el truco kernel.
Resolver un problema de optimización cuadrática que se ajuste a un hiperplano óptimo para clasificar las características transformadas en dos clases. El número de características transformadas está determinado por el número de vectores de soporte.

**Codigo**
Código

// Se importan todas la librerias necesarias
import org.apache.spark.ml.classification.LinearSVC

// Se cargan los datos en la variable "training" en un formato "libsvm"
val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// Se entrena el modelo con todos los datos del archivo
val lsvcModel = lsvc.fit(training)

// Se manda a imprimir los coefcientes de super vector machine
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

**Practica 7**

**Introducción**
En el siguiente documento hablaremos un poco de el algoritmo One-vs-Rest classifier y una pequeña aplicación en el lenguaje de programación scala.

**Explicación:**
OneVsRest es un ejemplo de una reducción de aprendizaje automático para realizar una clasificación multiclase dado un clasificador base que puede realizar la clasificación binaria de manera eficiente. También se conoce como "Uno contra todos".
OneVsRest se implementa como un Estimador. Para el clasificador base, toma instancias de Clasificador y crea un problema de clasificación binaria para cada una de las k clases. El clasificador para la clase i está entrenado para predecir si la etiqueta es i o no, distinguiendo la clase i de todas las demás clases.
Las predicciones se realizan evaluando cada clasificador binario y el índice del clasificador más seguro se genera como etiqueta.

**Código**
// Se importan todas la librerias necesarias
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Se cargan los datos en la variable "inputData" en un formato "libsvm"
val inputData = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")

// Se declararan 2 arreglos, uno tendra los datos de entrenamiento y el otro tendra
// los datos de prueba, respectivamente fueron declarados como arreglos y tendran el 80 y 20 porciento de los datos totales
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

// Se declara la variable "classifier" que hara la regresion
val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)

// Se declara el modelo "OneVsRest"
val ovr = new OneVsRest().setClassifier(classifier)

// Se entrena el modelo con los datos de entrenamiento
val ovrModel = ovr.fit(train)

// Se hacen las predicciones con los datos de prueba
val predictions = ovrModel.transform(test)

// Se declara el evaluador que tomara la precision del modelo y lo guardara en una variable metrica llamada "accuracy"
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

// Se calcula el error del modelo con una simple resta
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")

**Practica 8**

**Introducción**
En el siguiente documento hablaremos un poco de el algoritmo Naive Bayes y una pequeña aplicación en el lenguaje de programación scala.

**Explicacion:**
Naive Bayes
Los modelos de Naive Bayes son una clase especial de algoritmos de clasificación de Aprendizaje Automático, o Machine Learning, tal y como nos referiremos de ahora en adelante. Se basan en una técnica de clasificación estadística llamada “teorema de Bayes”.
Estos modelos son llamados algoritmos “Naive”, o “Inocentes” en español. En ellos se asume que las variables predictoras son independientes entre sí. En otras palabras, que la presencia de una cierta característica en un conjunto de datos no está en absoluto relacionada con la presencia de cualquier otra característica.

**Código**
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// Cargar los datos almacenados en formato LIBSVM como un DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
// Dividir los datos en conjuntos de entrenamiento y prueba (30% para pruebas)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
// Entrena un modelo NaiveBayes.
val model = new NaiveBayes().fit(trainingData)
// Seleccione filas de ejemplo para mostrar.
val predictions = model.transform(testData)
predictions.show()
// Seleccionar (predicción, etiqueta verdadera) y calcular error de prueba
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = $accuracy")

**Practica 9**

**Introducción**
La siguiente práctica es el resultado de la exposición 8 impartida por mis compañeros en la unidad 2 de la materia de datos masivos.

**Explicación**
Los clasificadores Naive Bayes son una familia de clasificadores probabilísticos y multiclase simples basados ​​en la aplicación del teorema de Bayes con fuertes supuestos de independencia (Naive) entre cada par de características.
Naive Bayes puede ser entrenado de manera muy eficiente. Con un solo paso sobre los datos de entrenamiento, calcula la distribución de probabilidad condicional de cada característica dada cada etiqueta. Para la predicción, aplica el teorema de Bayes para calcular la distribución de probabilidad condicional de cada etiqueta dada una observación.

**Código**
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// Cargar los datos almacenados en formato LIBSVM como un DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
// Dividir los datos en conjuntos de entrenamiento y prueba (30% para pruebas)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
// Entrena un modelo NaiveBayes.
val model = new NaiveBayes().fit(trainingData)
// Seleccione filas de ejemplo para mostrar.
val predictions = model.transform(testData)
predictions.show()
// Seleccionar (predicción, etiqueta verdadera) y calcular error de prueba
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = $accuracy")

# Evaluación 
**Examen Unidad 2 **
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

