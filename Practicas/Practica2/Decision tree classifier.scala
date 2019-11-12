
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
