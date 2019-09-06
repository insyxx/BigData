//primero
//Versión recursiva descendente
def fib (n: Int): Int = 
{
    if (n<2) {                                                 //si n<2 entonces
        return n                                               // devuelve n
    }
    else{		                                              //en otro caso	
        return fib(n-1) + fib(n-2)                                 //devuelve
    }
}


//segundo
//Versión con fórmula explícita
def fib2 (n: Double): Double ={
    if (n<2){                                                           //si n<2 entonces
        return n                                                        //devuelve n
    }
    else{                                                                 //en otro caso
        var p = ((1+(Math.sqrt(5)))/2)
        var j = (((Math.pow(p,n))-(Math.pow((1-p),n)))) /(Math.sqrt(5))
        return(j)                                                                   //deveulve j
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
    for(k <- 1 to n){                //para k desde 0 hasta n hacer

            b = b + a
            a = b - a
        }
        return(a)                   //deveulve a
}

//Quinto
//Version iterativa vector
def fib5 (n: Int): Double ={

    val vector = Array.range(0,n+1)

    if (n < 2){                               //si n<2 entonces
        return (n)                            // devuelve n
    }

    else{                                   // en otro caso
        vector(0) = 0
        vector(1) = 1

        for (k <- 2 to n){                   //Para k desde 2 hasta n+1 hacer
            vector(k) = vector(k-1) + vector(k-2)
        } 
        return vector(n)                      //devuelve vector n
    }
}
//Sexto
//Version Divide y Venceras 
def fib6 (n: Double): Double = {

    if (n<=0){                                // si n<=0 etncones
        return (n) 							  // devuelve 0
    }

    else{
        var i: Double = n - 1
        var auxOne: Double = 0
        var auxTwo: Double  = 1 
        var a: Double  = auxTwo
        var b: Double = auxOne
        var c: Double  = auxOne
        var d: Double  = auxTwo

        while (i > 0){                          //mientras i<0 hacer

            if (i % 2 == 1){ 					//si i es impar entonces

                auxOne = (d*b) + (c*a)
                auxTwo = ((d+(b*a)) + (c*b))
                a = auxOne
                b = auxTwo
            }

            else{								//si i es par entonces

                var pow1 = Math.pow(c,2)
                var pow2 = Math.pow(d,2)
                auxOne = pow1 + pow2
                auxTwo = (d*((2*(c)) + d))
                c = auxOne
                d = auxTwo
            }

            i = (i / 2)   
        }
        return(a+b) 							//deveulve a+b
    }
}

