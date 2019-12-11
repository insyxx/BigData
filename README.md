# Datos Masivos - Unidad 1

 - **Tecnológico Nacional de México**
 - **Instituto Tecnológico de Tijuana**
 - **Subdirección académica**
 - **Departamento de Sistemas y Computación**
 - **Semestre**: AGOSTO- DICIEMBRE 2019
 - **Ingeniería en Tecnologías de la Información y Comunicaciones**
 - **Materia**: Datos Masivos
 - **Unidad**: 1
 - **Nombre**: Rocha Zepeda Jaime
 - **No. Control**: 16211349
 - **Docente**: Dr. Jose Christian Romero Hernandez
 
 # Practicas
**Practica 1**

**Introducción**

En esta práctica aplicamos los principios de la programación en scala aprendidos durante los primeros días del curso

**Instruccion**
1.  Desarrollar un algoritmo en scala que calcule el radio de un circulo.
2.  Desarrollar un algoritmo en scala que me diga si un numero es primo.
3.  Dada la variable bird = "tweet", utiliza interpolacion de string para imprimir "Estoy ecribiendo un tweet".
4.  Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la secuencia "Luke".
5.  ¿Cual es la diferencia en value y una variable en scala?.
6.  Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el numero 3.1416.

**Código** 

Codigo

// Assessment 1/Practica 1

//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo

val c = 6 
var radio = c/(2*3.1416)

//2. Desarrollar un algoritmo en scala que me diga si un numero es primo

def esPrimo(i :Int) : Boolean = {
 	if (i <= 1)
   	false
 	else if (i == 2)
   	true
 	else
  	!(2 to (i-1)).exists(x => i % x == 0)
   }
esPrimo(2)
res4: Boolean = true
   
//3. Dada la variable bird = "tweet", utiliza interpolacion de string para
//   imprimir "Estoy ecribiendo un tweet"

var bird = "tweet"
println ("Estoy escribiendo un " + bird)

//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la
//   secuencia "Luke"

var variable = "Hola Luke soy tu padre!"
variable.slice(5,9)

//5. Cual es la diferencia en value y una variable en scala?
//Value (val) se le asigna un valor y ya no puede ser cambiado
//Variable (var) el valor asignado puede ser cambiado


//6. Dada la tupla ((2,4,5),(1,2,3),(3.1416,23)) regresa el numero 3.1416
var my_tup = (2,4,5,1,2,3,3.1416,23)
my_tup._7


**Practica 2**

**Introducción**
En esta práctica aplicamos los conocimientos is,else,for,List y range en esta práctica de scala.

**Instrucciones**
1. Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro".
2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla".
	Traer los elementos de "lista" "verde", "amarillo", "azul".
3.	Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5.
	Cuales son los elementos unicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversión a conjuntos.
4.	Crea una mapa mutable llamado nombres que contenga los siguiente "Jose", 20,"Luis", 24, "Ana", 23, "Susana", "27".
5.	Imprime todas la llaves del mapa.
6.	Agrega el siguiente valor al mapa("Miguel", 23).

**Código**

//Practice 2
// 1. Crea una lista llamad "lista" con los elementos "rojo", "blanco", "negro"

import scala.collection.mutable.ListBuffer
val lista = collection.mutable.ListBuffer("rojo","blanco","negro")
// 2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"
lista += "verde"
lista += "amarillo"
lista += "azul"
lista += "naranja"
lista += "perla"
// 3. Traer los elementos de "lista" "verde", "amarillo", "azul"
lista slice (3,6)
// 4. Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5
Array.range(1, 1000, 5)

// 5. Cuales son los elementos unicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversion a conjuntos
val lista = List(1,3,3,4,6,7,3,7)
lista.toSet

// 6. Crea una mapa mutable llamado nombres que contenga los siguiente
// 	"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"

val mutmap = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Susana", 27))

// 6 a . Imprime todas la llaves del mapa
println(mutmap)

// 7 b . Agrega el siguiente valor al mapa("Miguel", 23)
mutmap += ("Miguel" -> 23)

**Practica 3**

**Introducción**

En esta práctica utilizamos los conocimientos sobre scala para aplicarlos en el algoritmo matemático sucesión fibonacci

**Instrucciones**
1. Programar los 5 algoritmos de la sucesión de Fibonacci


**Explicación**
la sucesion de Fibonacci se realiza sumando siempre los ultimos 2 numeros (Todos los numeros presentes en la sucesion se llaman numeros de Fibonacci) de los siguientes 5 algoritmos matematicos:

**Código** 

//primero
//Versión recursiva descendente
def fib (n: Int): Int =
{
	if (n<2) {                                             	//si n<2 entonces
    	return n                                           	// devuelve n
	}
	else{   	                                           	//en otro caso    
    	return fib(n-1) + fib(n-2)                             	//devuelve
	}
}


//segundo
//Versión con fórmula explícita
def fib2 (n: Double): Double ={
	if (n<2){                                                       	//si n<2 entonces
    	return n                                                    	//devuelve n
	}
	else{                                                             	//en otro caso
    	var p = ((1+(Math.sqrt(5)))/2)
    	var j = (((Math.pow(p,n))-(Math.pow((1-p),n)))) /(Math.sqrt(5))
    	return(j)                                                               	//deveulve j
	}

}

//tercero
//Version iterativa
def fib3 (n: Int): Int ={
var a = 0
var b = 1
var c = 0
	for (k <- 1 to n){
    	c = b + a
    	a = b
    	b = c
	}
	return(a)
}

//Cuarto
//Version iterativa 2 variables
def fib4 (n: Int): Int ={
	var a = 0
	var b = 1
	for(k <- 1 to n){            	//para k desde 0 hasta n hacer

        	b = b + a
        	a = b - a
    	}
    	return(a)               	//deveulve a
}

//Quinto
//Version iterativa vector
def fib5 (n: Int): Double ={

	val vector = Array.range(0,n+1)

	if (n < 2){                           	//si n<2 entonces
    	return (n)                        	// devuelve n
	}

	else{                               	// en otro caso
    	vector(0) = 0
    	vector(1) = 1

    	for (k <- 2 to n){               	//Para k desde 2 hasta n+1 hacer
        	vector(k) = vector(k-1) + vector(k-2)
    	}
    	return vector(n)                  	//devuelve vector n
	}
}
//Sexto
//Version Divide y Venceras
def fib6 (n: Double): Double = {

	if (n<=0){                            	// si n<=0 etncones
    	return (n)    						   // devuelve 0
	}

	else{
    	var i: Double = n - 1
    	var auxOne: Double = 0
    	var auxTwo: Double  = 1
    	var a: Double  = auxTwo
    	var b: Double = auxOne
    	var c: Double  = auxOne
    	var d: Double  = auxTwo

    	while (i > 0){                      	//mientras i<0 hacer

        	if (i % 2 == 1){    				 //si i es impar entonces

            	auxOne = (d*b) + (c*a)
            	auxTwo = ((d+(b*a)) + (c*b))
            	a = auxOne
            	b = auxTwo
        	}

        	else{   							 //si i es par entonces

            	var pow1 = Math.pow(c,2)
            	var pow2 = Math.pow(d,2)
            	auxOne = pow1 + pow2
            	auxTwo = (d*((2*(c)) + d))
            	c = auxOne
            	d = auxTwo
        	}

        	i = (i / 2)   
    	}
    	return(a+b)    						 //deveulve a+b
	}
}

**Practica 4**

**Introducción**
En el siguiente código podremos ver algunas de las funciones más básicas de los data frame

**Instrucciones**
1. Agregar 20 funciones básicas para el la variable "df".

**Código**
//Codigo
//20 funciones de df
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.format("csv").option("header", "true").load("CitiGroup2006_2008")

"df"
//1
df.show()
//2
df.columns
//3
df.head()
//4
df.select("Date").show()
//5
df.printSchema()
//6
df.describe()
//7
df.sort()
//8
df.first()
//9
df.count
//10
df.filter($"Close" < 480 && $"High" < 480).show()
//11
df.select(corr("High", "Low")).show()
//12
df.select($"Close" < 500 && $"High" < 600).count()
//13
df.select(sum("High")).show()
//14
df.select(max("High")).show()
//15
df.select(min("High")).show()
//16
df.select(mean("High")).show()
//17
df.filter($"High"===484.40).show()
//18
df.filter($"High" > 480).count()
//19
df.select(year(df("Date"))).show()
//20
df.select(month(df("Date"))).show()

**Practica 5**

**Introducción**
En el siguiente código podremos ver algunas de las funciones más básicas de los data frame en agrupación de datos

**Instrucciones
1. Agregar 5 funciones 

//1 .Suma de columa 
df.select(sum("Volume")).show()

//2. Varianza
df.stat.cov("High", "Low")
//3. Minimo de columna
df.select(min("Volume")).show()

//4. Correlación 
df.stat.corr("High", "Low")

//5. Maximo de columna
df.select(max("Volume")).show()

**Practica 6**

**Introducción**
5 funciones utilizando el sintaxis de Sparksql y Scala

**Instrucciones**
Agregar 5 funciones utilizando el sintaxis de Sparksql y Scala

**Código**
//1
df.filter($"Close">480).show()

// 2
df.filter($"Close" < 480 && $"High" < 480).show()

df.filter("Close < 480 AND High < 480").show()
// 3
df.select(month(df("Date"))).show()

df.select(year(df($"Date"))).show()

val df2 = df.withColumn("Year", year(df("Date")))

// 4
val dfavgs = df2.groupBy($"Year").mean()

dfavgs.select($"Year", $"avg(Close)").show()
// 5
val dfmins = df2.groupBy($"Year").min()

dfmins.select($"Year", $"min(Close)").show()

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
