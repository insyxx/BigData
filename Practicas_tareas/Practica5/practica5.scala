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