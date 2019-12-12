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