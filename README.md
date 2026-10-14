#   
# Lab 3.4: Training a Machine Learning Model  
  
![=AMUFV1ES-L1113-179172 > Modulos > Miduo 3 - Inplementación de un.. > Laboopone 3.4 - Amazon SageMaker s](pantallazos/EAEE8230-31CC-4F0E-8F5C-6FA6D41531B8.png)  
  
  
  
Seleccionear el servicio :   Amazon SageMaker AI  
  
![Pasted Graphic 36.png](pantallazos/3B08F4D2-A32B-4879-838C-B6C137668A4B.png)  
  
  
  
Seleccionamos notebook  
  
![Pasted Graphic 38.png](pantallazos/359E60A1-A03D-4965-A0F5-5B175E609D96.png)  
  
  
  
  
  
  
  
  
Revisamos los archivos   
  
Inicializamos el reposo  
  
Crear carpeta pantallazos  
Inicializar el repositorio de git   
  
Configurar usuario GitHub  
  
  
Configuration previa  en el repositorio  
  
![Pasted Graphic 43.png](pantallazos/752197CD-325C-4D50-B774-9295E914D62B.png)  
  
  
  
Luego  
Crear git branch   dev_jurbano18  
  
Realizamos el primer comió con los archivos actuales  
3_4-machinelearning.ipynb  
PythonCheatSheet.ipynb  
  
Primer commit “initial commit”  
![Pasted Graphic 44.png](pantallazos/3FAC2D5E-5848-4579-9296-6811AA75C401.png)  
  
  
  
Instalamos dependencias  scipy  
  
  
![Pasted Graphic 41.png](pantallazos/94697C86-EC7A-4CD5-ACB3-CEDC27E1674C.png)  
  
  
Importar dependencias  
  
descargar el dataset de machine-learning-databases  
  
Al lado derecho se ven los archivos  
  
![Contiguración del laboratorio](pantallazos/D74AF8B1-F1E4-4ED4-BDAE-7B8176484EF8.png)  
  
  
Se descarlos  los archivos en el directorio actual   
  
column_2C.dat  
column_2C_weka.arff    
column_3C.dat    
column_3C_weka.arff  
  
![Pasted Graphic 45.png](pantallazos/32929DAD-6834-4226-BD14-B8BF72AFE19F.png)  
  
  
  
Proceso de importación  
  
# descargar el dataset de machine-learning-databases  
  
# se encarga de cargar un archivo de datos (column_2C_weka.arff)   
# en formato ARFF y transformarlo en una tabla de tipo dataframe  
  
  
# Crea un diccionario de mapeo que relaciona los valores de texto originales Abnormal, Normal   
# con los número 1, 0  
  
  
# Aplica ese diccionario sobre la columna llamada class del DataFrame (df)  
  
Para que se  
  
El algoritmos de aprendizaje XGBoost no pueden entrenarse directamente utilizando etiquetas de texto así que necesitan que las clases estén representadas en formato numérico binario  
  
![Pasted Graphic 46.png](pantallazos/121B4E4E-3956-494F-9B52-0AC13004AE74.png)  
La diferencia entre ambos estados del DataFrame es la conversión de la etiqueta de clase de texto a formato numérico en la columna class  
  
  
  
  
paso 1 Exploramos los datos  
  
# conocer las dimensiones de un DataFrame, devuelve el número de columnas y filas  
  
#  muestra las columnas del dataframe  
  
![Paso 2: Preparar los datos](pantallazos/C4D78A0A-86F7-490A-8C4D-A83DC2B53F7A.png)  
  
  
En esta parte antes de pasar a la preparación y división de los datos lo que se hace es que garantiza que la estructura del conjunto de  datos cargado coincida con lo esperado por el modelo  
  
  
  
Ahora subimos al repositorio los archivos descargados del dataset y cargamos los comentarios y actualización del laboratorio 3_4-machinelearning.ipynb  
![Pasted Graphic 48.png](pantallazos/1B764AC1-DEC9-44F9-A250-4F251A0B852E.png)  
  
  
  
  
Paso 2: Preparar los datos  
  
# vemos como estaba antes de organizar el dataframe, la columna class de última columna  
  
# luego se reorganizar las columnas del DataFrame, moviendo  la columna de clase a la primera posición   
# la columna de más a la izquierda  
  
# finalmente se muestra las columnas del dataframe donde muestra class como la primera columna  
  
¿Para que esto? Cuando se utiliza **XGBoost en Amazon SageMaker** al utilizar el formato de datos CSV requiere que la columna class que el modelo debe predecir se encuentre en la **primera columna** del archivo.   
Si está en otra posición entonces el algoritmo la interpretará mal  
  
![salción de la columna objetivo](pantallazos/2EDDFEE1-7B3E-4D2F-A1AA-8B3283D8414E.png)  
  
  
  
  
División de los datos  
  
Aquí empezamos por instalar la dependencia  *scikit-learn*  
![Pasted Graphic 50.png](pantallazos/1E59F033-AD4C-445B-8C42-26B0204B8F66.png)  
  
  
A continuación, divida el conjunto de datos *probar_y_validar* en dos partes iguales.  
  
  
  
Para qué   
  
Realiza una **división **de los datos donde garantiza que ambas mitades conserven  la misma proporción equilibrada de pacientes sanos y enfermos y asegurando la reproducirlos del experimento mediante la semilla  
  
Al final tuvo como resultado  
  
**Conjunto de entrenamiento (train)**: 168 pacientes anormales (1) y 80 pacientes normales (0).  
**Conjunto de prueba (test)**: 21 pacientes anormales (1) y 10 pacientes normales (0).  
**Conjunto de validación (validate)**: 21 pacientes anormales (1) y 10 pacientes normales (0).  
  
  
  
  
![Pasted Graphic 51.png](pantallazos/6606290E-7F7D-4967-9295-2FFFEBBE588C.png)  
  
  
  
Carga de los datos a Amazon S3  
  
  
# convertir DataFrames a formato CSV y subirlos directamente a la ruta del bucket en Amazon S3  
  
¿Para qué?  
 XGBoost de AWS no lee archivos locales en el laboratorio  sino que es requisito que los datos estén almacenados  en la nube (S3) en formato CSV plano sin encabezados ni índices para evitar saturar el disco virtual con archivos temporales innecesarios  
  
  
   
![Pasted Graphic 52.png](pantallazos/3FE3A3B9-7437-4678-8080-2AEF3F1E7C68.png)  
  
  
  
  
  
  
Luego se cargan en el bucket s3  
![Pasted Graphic 53.png](pantallazos/430ABD61-54A5-4324-9EDF-04BED0F706DD.png)  
  
3 carpetas test, train y validate  
![Pasted Graphic 54.png](pantallazos/479A8E48-D9CE-4438-B868-B7E0CDA649D2.png)  
  
En cada carpeta hay un archivo llamado  
* vertebral_train.csv  
* vertebral_test.csv  
* vertebral_validate.csv  
  
![Pasted Graphic 55.png](pantallazos/757280A7-04B0-4024-96B6-5327F21FA947.png)  
  
  
  
Subo los cambios al reposiorio hasta la carga de datos s3  
  
![Pasted Graphic 56.png](pantallazos/830C26A8-8D6C-421B-84E4-F558F06FEA51.png)  
  
  
  
  
  
Paso 3: Entrenar el modelo  
  
  
Para qué?  
Le indica al servicio SageMaker qué motor de aprendizaje automático preconfigurado debe preparar y descargar para entrenar tu modelo  
  
  
Esto se hace para **crear y configurar el "entrenador" **que utilizará Amazon SageMaker para construir el modelo  
* **Utiliza:** El algoritmo XGBoost que preparado antes.  
* **Dónde guardar el resultado:** En una carpeta específica de tu S3 (output)  
* **Qué equipo usar:** Una computadora virtual potente en la nube  
  
  
![Pasted Graphic 23.png](pantallazos/78AFEF62-FB1E-41A3-8230-FBD0E003E20B.png)  
  
  
Finalmente  
  
Se conectan los archivos de entrenamiento y validación que subiste a S3 para que la nube sepa de dónde sacar la información  
  
Al ejecutar esta orden, la computadora virtual en la nube descarga los datos y luego procesa el aprendizaje de XGBoost y muestra el avance paso a paso  
  
![Traising is progress...](pantallazos/AFF2A358-428F-475A-B804-42A2FC5BE273.png)  
  
