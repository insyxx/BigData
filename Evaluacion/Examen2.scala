var scores = Array(10,5,20,20,4, 5, 2, 25, 1)
var scores = Array(3,4,21,36,10,28,35,5,24,42)

def breakingRecords(scores:Array[Int]):=   //se inicia la funcion donde tomara un arreglo tipo entero
{
    var score = 0           // Se agrega un contador para recorrer los espacios del arreglo
    var max = scores.head   // Se asigna el primer valor de arreglo a una variable
    var min = scores.head

    var higherScore = 0     //En estas variables se iran asigna los valores de los puntajes
    var lowerScore = 0

    for(score <- scores)   //El contador ira pasando por los espacios del arreglo
    {
        if (score > max) 
        {                  //si el score es mayor ala primer numero del arreglo
            max = score                     //iguala la variable max a score
            higherScore = higherScore + 1   //entonces se le suma uno al puntaje alto
        }

        if (score < min) 
        {
            min = score
            lowerScore = lowerScore + 1
        } 
    }
    println("Puntajes altos: " + higherScore + " " + "Puntajes bajos " + lowerScore)   //imprime las veces que tuvo puntajesaltos y bajos
}