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
