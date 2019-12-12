# Evaluación 
**Examen 1**

**Introducción**

Problema
Maria juega baloncesto universitario y quiere ser profesional, cada temporada ella mantienen su registro de su juego, tabula la cantidad de veces que rompe su record de temporada para la mayoría de los puntos y la menor cantidad de puntos en un
juego. Los puntos anotados en el primer juego establece su récord para la temporada, y ella comienza a contar desde alli.
Por ejemplo, suponga que sus puntajes para la temporada están representados en la matriz. Las puntuaciones están en el mismo orden que los juegos jugados. Ella tabularia sus resultados de la siguiente manera: Score = [12,42,10,24]

Teniendo en cuenta los puntajes de Maria para una temporada, encuentra e imprime  el número de veces que rompe sus récords para la mayoría y la menor cantidad de
puntos anotados durante la temporada.
Sample Input
9
10, 5, 20, 20, 4, 5, 2, 25, 1
Sample Output
2 4

**Codigo**
val scores = List(10,5,20,20,4, 5, 2, 25, 1)
val scores = List(3,4,21,36,10,28,35,5,24,42)

def breakingRecords(scores:List[Int]):Unit={    //se inicia la funcion donde tomara un arreglo tipo entero
    var score = 0                               // Se agrega un contador para recorrer los espacios del arreglo
    var max = scores.head                       // Se asigna el primer valor de arreglo a una variable
    var min = scores.head

    var higherScore = 0                         //En estas variables se iran asigna los valores de los puntajes
    var lowerScore = 0

    for(score <- scores){                        //El contador ira pasando por los espacios del arreglo
        if (score > max){                       //si el score es mayor ala primer numero del arreglo
            max = score                         //iguala la variable max a score
            higherScore = higherScore + 1       //entonces se le suma uno al puntaje alto
        }

        if (score < min) {
            min = score
            lowerScore = lowerScore + 1
        } 
    }
    println("Puntajes altos: " + higherScore + " " + "Puntajes bajos " + lowerScore)   //imprime las veces que tuvo puntajesaltos y bajos
}

breakingRecords(scores)

**Examen 1.2**

**Introducción**

**Codigo**

//1. Comienza una simple sesion spark
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

//2. Cargue el archivo Netflix Stock CSV, haga que Spark infiera los tipos de datos
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

//3. Cuales son los nombres de las columnas
df.columns

//4. Como es el esquema
df.printSchema()

//5. Imprime las primeras 5 columnas
df.head(5)

//6. Usa describe () para aprender sobre el dataframe
df.describe().show()

//7. Crea un nuevo data frame con una columna llamada "HV Ratio" 
//que es la relacion entre el preio de la columna "High" frente de ala 
//columna "Volumen" de acciones negociadas por una dia
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df2.show()

//8.

//9. Cual es el significado de la columna Cerrar "Close"
//La cantidad con la que cerro el mes
//10. Cual es el maximo y minimo de la columna volumen
df.select(max("Volume")).show()
df.select(min("Volume")).show()

//11. Con sintaxis Scala/Spark $ conteste lo siguiente

import spark.implicits._ //importa variables implicitas de spark

//a. Cuantos dias de la columna "Close" inferior a $600
df.filter($"Close"<600).count()
//b. Que porcentaje del tiempo fue la columna "High" mayor que $500
(df.filter($"High" > 500).count() * 1.0/ df.count())*100
//c. Cual es la correlacion de pearson entre la columna "High" y "Volume"?
df.select(corr("High","Volume")).show()

//d. Cual es maximo de la columna "High" por ano
val aniodf = df.withColumn("Year",year(df("Date"))) //Agrega columna ano, extrayendola de la columa Date
// Se selecciono a partir de la variable "maximoanio" el año y el maximo de "High", a partir de maximos a minimo en los años
val maximoanio = aniodf.select($"Year",$"High").groupBy("Year").max() 
// Se dio como resultado el maximo de la columna "High" por año
val dfres = maximoanio.select($"Year",$"max(High)")
dfres.orderBy("Year").show()

//e. Cual es el promedio de la columna "Close" para cada mes del calendario
val mesdf = df.withColumn("Month",month(df("Date"))) // Agrega columna mes, extrayendola de la columna date 
// Se selecciono a partir de la variable "mesdf" el mes y el promedio de "Close"
val mesavg = mesdf.select($"Month",$"Close").groupBy("Month").mean()
// Se selecciono y mostro el promedio de la columna "Close" para cada mes del calendario
mesavg.select($"Month",$"avg(Close)").orderBy("Month").show()
