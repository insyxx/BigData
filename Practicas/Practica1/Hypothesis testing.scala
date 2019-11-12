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