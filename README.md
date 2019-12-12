# Datos Masivos - Unidad 4

 - **Tecnológico Nacional de México**
 - **Instituto Tecnológico de Tijuana**
 - **Subdirección académica**
 - **Departamento de Sistemas y Computación**
 - **Semestre**: AGOSTO- DICIEMBRE 2019
 - **Ingeniería en Tecnologías de la Información y Comunicaciones**
 - **Materia**: Datos Masivos
 - **Unidad**: 4
 - **Nombre**: Rocha Zepeda Jaime
 - **No. Control**: 16211349
 - **Docente**: Dr. Jose Christian Romero Hernandez
 
 #Proyecto Unidad 4

**Introducción**

En el siguiente documento se presentan 3 tipos de algoritmos de tipo machine learning los cuales se ejecutará una pequeña aplicación de cada uno para dar a conocer cuál puede ser más precioso.

**Marco teórico de los algoritmos**
**Árbol de decisión**
Un Árbol de decisión (también llamados árbol de toma de decisiones, árbol de decisión o árboles de decisiones) es un esquema que representa las alternativas disponibles para quien va a tomar la decisión, además de las circunstancias y consecuencias de cada elección. Su nombre proviene del aspecto similar a un árbol y sus ramificaciones que tiene este diagrama.

Los árboles de decisiones están conformados por una serie de nodos de decisiones con ramas que llegan y salen de ellos. Estos nodos pueden ser:
Nodos Cuadrados o de decisión: Representan los puntos de decisión donde se muestran las distintas alternativas disponibles a elegir. Se escoge la alternativa que presenta el mayor valor esperado.
Nodos Circulares o de probabilidad: Donde salen las diferentes ramificaciones que muestran los hechos fortuitos que tienen una probabilidad de ocurrencia. La suma de las probabilidades de cada suceso (rama) que sale de un nodo circular debe ser uno. El valor esperado del nodo se obtiene realizando un promedio ponderado de las ramificaciones con sus probabilidades.
Nodos Terminales: Representan un resultado definitivo de una ramificación.
Las ramificaciones se representan de la siguiente forma:
Ramificaciones alternativas: Cada ramificación representa un resultado probable.
Alternativa rechazada: Una vez desarrollado el árbol, las alternativas que no se seleccionan se marcan con dos líneas.





**Ejemplo árbol de decisión:**

![](https://lh6.googleusercontent.com/OtYLiE3UhiceSYwuE34VYDNuncrydKe-mxTMe4jT6Bi_4fhHCnCKO76fy-T9nw62aE5AV8i9TnsJ4rxubo97En-_dddq8S5euTTzx2zk2QynaEdiY7sHVRODD7EQ1wU7IIlzOVQe)

**Regresión logística**

La regresión logística es un tipo de análisis de regresión utilizado para predecir el resultado de una variable categórica (una variable que puede adoptar un número limitado de categorías) en función de las variables independientes o predictoras. Es útil para modelar la probabilidad de un evento ocurriendo como función de otros factores. El análisis de regresión logística se enmarca en el conjunto de Modelos Lineales Generalizados que usa como función de enlace la función logit. Las probabilidades que describen el posible resultado de un único ensayo se modelan, como una función de variables explicativas, utilizando una función logística.


 
 
 
 
 
 
 
**Perceptrón multicapa**

El perceptrón multicapa evoluciona el perceptrón simple y para ello incorpora capas de neuronas ocultas, con esto consigue representar funciones no lineales.
El perceptrón multicapa está compuesto por por una capa de entrada, una capa de salida y n capas ocultas entremedias.
Se caracteriza por tener salidas disjuntas pero relacionadas entre sí, de tal manera que la salida de una neurona es la entrada de la siguiente.
En el perceptrón multicapa se pueden diferenciar una 2 fases:
Propagación en la que se calcula el resultado de salida de la red desde los valores de entrada hacia delante.
Aprendizaje en la que los errores obtenidos a la salida del perceptrón se van propagando hacia atrás (backpropagation) con el objetivo de modificar los pesos de las conexiones para que el valor estimado de la red se asemeja cada vez más al real, este aproximación se realiza mediante la función gradiente del error.

**Arquitectura del perceptrón multicapa:**

![](https://lh3.googleusercontent.com/5LEhnfZYwazv6i1NmWlSfXEuxRA4NfvIVxaG3jEGXlwwI1ogIMI-8YP733xir3y2OBg8EXMwnMry1WrvhxelVSPXtvE3v8bxJgA-GvOzgBVi9SH-3arVuPMhy4X0luyQs48m6qqE)



**Implementación**

El software utilizado para realizar las pruebas fue Apache spark el cual es un sistema informático de alto rendimiento open source.
Spark permite al usuario trabajar con el lenguaje scala el cual fue desarrollado con el objetivo de ser un lenguaje de alto rendimiento y altamente concurrente y combine fuerza con uno de los líderes en programación en plataforma Java Virtual Machine.

Spark es flexible en su utilización, y ofrece una serie de APIs que permiten a usuarios con diferentes backgrounds poder utilizarlo. Incluye APIs de Python, Java, Scala, SQL y R, con funciones integradas y en general una performance razonablemente buena en todas ellas.
Permite trabajar con datos más o menos estructurados (RDDs, dataframes, datasets)
dependiendo de las necesidades y preferencias del usuario.


**Dataset (bank-full)**

Los datos están relacionados con campañas de marketing directo de una institución bancaria portuguesa. Las campañas de marketing se basaron en llamadas telefónicas. A menudo, se requería más de un contacto con el mismo cliente para acceder si el producto (deposito bancario a plazo) estaría ('sí') o no ('no') suscrito.

Tomamos todos los ejemplos (41188) y 11 entradas, ordenadas por fecha (de Mayo de 2008 a noviembre de 2010).
De los datos a trabajar, usamos dos categorías de la información proveída y tomamos los siguientes:

Información bancaria del cliente
age números
job: valores categóricos convertidos a números
marital: estado civil categórico convertido a números
education: nivel de estudios categóricos convertido a números
default: si cuenta con algún tipo de crédito, datos categóricos convertidos a números
housing: si renta casa, numerico
loan: si tiene algún crédito personal, numérico



Información del cliente relacionada a la campaña actual y el último contacto

campaign: cuantas veces se ha contactado al cliente para la campaña actual
days: cuántos días han pasado desde el último contacto  de la campaña
previous: cuantas veces se ha contactado al cliente anteriormente
outcome: cómo han sido los contactos anteriores.



**Resultados**
Utilizando los algoritmos anteriormente nombrados, se realizaron varias pruebas con el conjunto de datos antes mencionado.

Se realizaron 15 pruebas , dado que la distribución de datos se realiza de manera aleatoria. Es decir, en cada corrida de las pruebas los 41188 registros se distribuyeron en un 70% para aprendizaje y 30% para pruebas. De esta manera en cada iteración  se distribuye en datos diferentes. 

**Tabulador de precisión de los 3 algoritmos**

![](https://lh5.googleusercontent.com/uYj6ktgIIIAL2tdodBM3Xu0A8G8l3mY1hLM0vnW1ZOXoYzdTi-n6XO7VpoKWtbe7rci4TykYuRvek4FUjMYmsPGfBv2adLzS5Bi0F7CeCUOF4T6hB8Uuuaizxv20_qw2yiaipj12)



**Nivel de precisión de los algoritmos**


![](https://lh3.googleusercontent.com/EsMX8tv-6Km5Hh_n2F35S49ZNXN5TypEEMxs3Tk9wHd4S0GoEvM_N9CiwuR1ld-4DIOYlj0yqlJy55XqX9XJPqOcqI25n5X3bOQR_PSvv0skd-zq1AtBXyubHOj3ijadJ6kdv6wS)



**Conclusión**

En conclusión el árbol de decisiones considero que fue uno de los algoritmos más precisos al dar exactitud en los datos, el cual también provee un esquema de los datos los cuales son fáciles de entender y permiten visualizar gráficamente las diferentes opciones para una existente a la hora de tomar una decisión 


