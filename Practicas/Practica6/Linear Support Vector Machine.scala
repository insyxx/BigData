// Se importan todas la librerias necesarias
import org.apache.spark.ml.classification.LinearSVC

// Se cargan los datos en la variable "training" en un formato "libsvm"
val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// Se entrena el modelo con todos los datos del archivo
val lsvcModel = lsvc.fit(training)

// Se manda a imprimir los coefcientes de super vector machine
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")