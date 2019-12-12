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