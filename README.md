
# Integrantes: Vicente Arechavala, Johan Riveros

**Problemática**  
El reconocimiento automático de gestos de manos es una tarea relevante dentro de la visión computacional, ya que puede utilizarse en interfaces sin contacto, videojuegos, sistemas educativos y aplicaciones de interacción entre personas y computadores. Sin embargo, reconocer correctamente un gesto mediante una imagen puede ser difícil debido a factores como:
* Diferencias en la posición y tamaño de la mano.
* Variaciones en la iluminación.
* Orientación de los dedos.
* Similitudes visuales entre algunos gestos.

**Pregunta de Investigación**  
¿Es posible clasificar correctamente imágenes de gestos de manos correspondientes a piedra, papel y tijera, manteniendo un buen desempeño frente a variaciones en la posición, orientación e iluminación de la mano?

**Descripción del Dataset**  
El modelo es descargado de la plataforma Kaggle, de nombre Rock, Paper, Scissors Dataset subido por Roman Glushko, en donde se muestran distintas fotografías tomadas presuntamente por el autor y su colaborador, Anastasia Smirnova. 

El dataset cuenta con imágenes de cada una de las opciones del tradicional juego de piedra, papel o tijeras, cada una de las opciones distribuídas en sus respectivas carpetas de nombre de la jugada perteneciente al juego (rock, paper y scissors respectivamente). El dataset ya venía con las carpetas diferenciadas, es decir, ya vienen con un grupo de imágenes para entrenamiento, otro para la validación, y un último para el testeo.

Como se dijo previamente, el dataset contiene imágenes de gestos de manos correspondientes al juego piedra, papel o tijera, capturadas mediante una cámara web.
* Rock: Mano cerrada formando una piedra.
* Paper: Mano abierta representando papel.
* Scissors: Dedo índice y medio extendidos formando una tijera.

La idea a aplicar con este dataset es utilizar un modelo preentrenado para poder analizar si es que dicho modelo puede identificar los patrones más relevantes para saber diferenciar las distintas imágenes del dataset, de forma que se identifiquen correctamente las opciones pertenecientes al juego, así como también probar la efectividad del modelo para dicho proposito.

**Justificación del modelo**  
ResNet18  

Para este proyecto se seleccionó ResNet18 como arquitectura principal de Deep Learning, dado que es una red con suficiente capacidad de aprendizaje para la detección de las diferencias entre los 3 diferentes estilos de manos provenientes del dataset. Además, dada las limitaciones actuales de las computadoras de quienes realizan el trabajo, la utilización de un modelo más exigente (como el visto en clases, resnet50) implica mayores costos computacionales, lo cual no necesariamente garantiza mejores resultados.

El objetivo es clasificar imágenes correspondientes a los gestos de piedra, papel y tijera, un problema de clasificación multiclase donde las diferencias entre categorías dependen de características visuales como la forma de la mano, la posición de los dedos y el contorno del gesto. ResNet18 ya posee la capacidad de reconocer patrones visuales generales, como bordes, líneas, texturas y formas. Mediante Transfer Learning, este conocimiento previo puede adaptarse al reconocimiento de los tres gestos del dataset, reduciendo significativamente el tiempo de entrenamiento y la cantidad de datos necesarios para obtener un buen desempeño.

* Ventajas: Presenta un buen equilibrio entre precisión y costo computacional siendo mas liviana que otros modelos, es una arquitectura ampliamente utilizada y validada en tareas de clasificación de imágenes, lo que garantiza estabilidad y disponibilidad de implementaciones optimizadas y posee una buena capacidad de generalización cuando se combina con técnicas de regularización y aumento de datos.
  
* Limitaciones: Aunque ResNet18 ofrece un buen rendimiento, su capacidad de representación es menor que la de arquitecturas más profundas, por lo que podría presentar dificultades frente a problemas con una gran cantidad de clases o patrones extremadamente complejos. Su desempeño depende de que las imágenes utilizadas, se espera que representen adecuadamente las variaciones presentes en un escenario real, como cambios de iluminación, posición o fondo.
  
* Pertinencia: La elección de ResNet18 es pertinente porque el problema consiste en reconocer patrones visuales bien definidos dentro de imágenes y clasificarlos en una de tres categorías. El modelo ofrece una combinación adecuada entre precisión, velocidad de entrenamiento y consumo de recursos computacionales, lo que lo convierte en una alternativa apropiada.

### Primeras Impresiones (Carga del dataset en adelante)

Se observa una cantidad importante de imágenes en la carpeta "val" y "test". A pesar de que la cantidad de imágenes en la carpeta "train" es mayor, normalmente debería seguir una distribución de sobre el 70% de las imágenes, cosa que no se cumple en este caso, ello pudiendo generar problemas para el modelo preentrenado para predecir correctamente las formas de las palmas de las manos. Aun así, se ha decidido no cambiar la distribución original prevista por el dataset, pues de esta manera se podrá evaluar de manera íntegra los resultados obtenidos, de forma que se pueda evaluar si llega a existir esta problemática o no con las configuraciones actuales del dataset.

<img width="593" height="398" alt="Cantidad de papel, piedra y tijeras" src="https://github.com/user-attachments/assets/5b876159-e443-4ab2-95cc-3344c248d615" /> 

Antes del entrenamiento se analizó la cantidad de imágenes por clase. Se observa que existe una distribución ligeramente mayor para las imágenes que contienen "papel" respecto de las otras opciones, que tienen una distribución exactamente igual. Esto, sin embargo, no necesariamente representa una problemática, pues representa un desbalance bajo, por lo que no se aplicarán técnicas de reducción de dicho desbalance. 

<img width="874" height="294" alt="Piedra" src="https://github.com/user-attachments/assets/4114a51b-feab-49c3-aecc-f7f6fc2feffd" />
<img width="874" height="284" alt="Papel" src="https://github.com/user-attachments/assets/f8b6bb5b-9516-4b56-8f29-e764d8b1d76d" /> 
<img width="874" height="290" alt="Tijeras" src="https://github.com/user-attachments/assets/2bbb1af6-b3c3-4b5c-95e9-d5f753637cfd" /> 

Se observan las distintas imágenes que existen para las 3 opciones que tiene el dataset. Como se puede ver en la figura, las imágenes del dataset no cuentan con una estructura específica, es decir, más allá de las 3 opciones permitidas, los diseños mostrados en cámara no siguen un patrón específico de distancia o posición, lo que da mayor diversidad en los patrones que debe determinar el dataset como pertenecientes a cierto tipo de características que deriven en la opción correcta o incorrecta (determinar correctamente si se trata de una piedra, papel, o tijera).

## Modelo

**Justificación del criterio de entrenamiento**  
Cross entropy  

Para el entrenamiento del modelo se utilizará la función de pérdida Crossentropy, ya que el problema corresponde a una clasificación multiclase, donde cada imagen pertenece exclusivamente a una de las tres categorías disponibles: piedra, papel o tijera. La función Crossentropy compara la distribución de probabilidades generada por el modelo con la etiqueta verdadera de cada imagen, ajusta sus parámetros para aumentar progresivamente la probabilidad de la categoría correcta y disminuir la de las demás.  

* Ventajas: Es la función de pérdida estándar para problemas de clasificación multiclase. Funciona de forma eficiente y penaliza con mayor intensidad las predicciones incorrectas realizadas con alta confianza, mejorando el proceso de aprendizaje.
  
* Limitaciones: Puede verse afectada cuando existe un fuerte desbalance entre las clases, es sensible a etiquetas incorrectas o mal clasificadas dentro del dataset.
  
* Pertinencia: Crossentropy constituye la elección más adecuada. Su utilización permite optimizar directamente las probabilidades generadas por el modelo y es la función de pérdida recomendada para problemas de clasificación supervisada de este tipo.

**Justificación del optimizador**  
AdamW  

Se ocupará una variante del optimizador Adam que incorpora una estrategia de Weight Decay desacoplada, permitiendo un mejor control de la regularización de los pesos del modelo. 

* Ventajas: Ha demostrado un excelente rendimiento en tareas modernas de visión computacional y clasificación de imágenes, ajusta automáticamente la tasa de aprendizaje para cada parámetro del modelo, acelerando la convergencia e incorpora Weight Decay, ayudando a controlar el crecimiento excesivo de los pesos (los pesos son números que determinan la importancia o la fuerza de la conexión entre diferentes neuronas de un modelo) y disminuyendo el sobreajuste.
  
* Limitaciones: Puede demandar un mayor tiempo de ajuste inicial para encontrar la configuración óptima, un Weight Decay (técnica de regularización que resta directamente un pequeño porcentaje al valor de los pesos en cada paso del entrenamiento) excesivo puede provocar subajuste (underfitting), reduciendo la capacidad del modelo para aprender patrones relevantes.
  
* Pertinencia: AdamW resulta especialmente adecuado para este proyecto porque combina rapidez de convergencia con una estrategia efectiva de regularización. Dado que el modelo ResNet18 será ajustado mediante Transfer Learning, AdamW permite optimizar los parámetros de manera eficiente mientras controla el sobreajuste, favoreciendo un mejor desempeño sobre imágenes no vistas durante el entrenamiento.

**Justificación Metricas**  
En este proyecto se utilizaron las métricas precision, recall, F1-score y support porque permiten evaluar con mayor detalle el desempeño del modelo ResNet18 al clasificar imágenes en las categorías piedra, papel y tijera. Estas métricas son más informativas que observar únicamente la exactitud general, ya que permiten analizar el comportamiento del modelo en cada clase por separado.  

* Recall: El recall indica qué proporción de las imágenes reales de una clase fueron reconocidas correctamente por el modelo. Por ejemplo, permite conocer cuántas de todas las imágenes reales de “piedra” fueron identificadas correctamente como piedra. Esta métrica es importante porque muestra si el modelo está dejando pasar ejemplos de alguna categoría.  

* Precisión: La precision indica qué proporción de las imágenes que el modelo predijo como una clase realmente pertenecían a ella. Por ejemplo, si el modelo clasifica varias imágenes como “tijera”, esta métrica permite saber cuántas de esas predicciones eran efectivamente tijeras. Se utiliza en este proyecto porque permite detectar si el modelo está confundiendo una clase con las otras.  

* F1-Score: El F1-score combina precision y recall en una sola medida. Se utiliza para obtener una evaluación equilibrada del rendimiento del modelo en cada clase. Es especialmente útil porque un modelo podría tener una precision alta, pero un recall bajo, o viceversa. El F1-score permite identificar si el modelo mantiene un buen equilibrio entre evitar predicciones incorrectas y reconocer correctamente la mayor cantidad posible de imágenes de cada categoría.  

* Support: El support representa la cantidad real de imágenes pertenecientes a cada clase dentro del conjunto de evaluación. Esta métrica no mide directamente el rendimiento del modelo, pero permite interpretar correctamente las demás métricas.    

## Resultados  

### _Resultados de las iteraciones de EPOCH se encuentran en el código_

Durante los Epoch, se observa que la pérdida de entrenamiento fue constante (pasó de 1.1495 a 0.3455), al igual que la pérdida de validación (1.0645 a 0.6179). Al mismo tiempo el accuracy de entrenamiento aumentó de 0.372 a 0.896, de la misma forma ocurre con el accuracy de validación, que aumentó de 0.430 a 0.749.
Estos resultados determinan que el modelo "aprendió" patrones que le resultan útiles para la clasificación de las imágenes, y dicho aprendizaje se estaba trasladando al conjunto de validación. No se activó el early stopping en ninguna de las iteraciones, porque el modelo seguía mejorando.

<img width="1289" height="490" alt="cce578f9-5233-4107-9a8e-e9de669c76ad" src="https://github.com/user-attachments/assets/94755c5b-e495-4916-9426-2d1baff43b51" />﻿# Proyecto-2---Introduccion-a-la-IA

Tal como se observó anteriormente, las pérdidas de entrenamiento y validación, así como las mejoras en el accuracy de los mismos tienen cierta consistencia. Dadas las leves mejoras en el crecimiento del accuracy para la validación desde el Epoch 10 en adelante (mayormente observable con los valores obtenidos en el ejercicio anterior) vemos que es posible que el modelo esté experimentando de overfitting. Sin embargo, las iteraciones realizadas aún conservan una diferencia entre entrenamiento y validación (diferencia entre 0.896 y 0.749), por lo que se puede argumentar que es un overfitting leve o controlado. Es menester que en caso de realizar mayor cantidad de iteraciones se observe dicha diferencia, porque es posible que en más de 20 iteraciones si se presente un caso de overfitting mayor al que se muestra actualmente.

En vista lo anterior, se considera que el modelo es apto para el siguiente paso de testeo.

### _Reporte de clasificación se encuentra en el código_

Se observa un accuracy del 0.5926 para el test, y si bien podría considerarse un valor relativamente bajo, pues el accuracy obtenido en la validación fue de 0.749, esto puede ocurrir dada la posibilidad de que las imágenes de la carpeta "test" contengan carácterísticas ligeramente distintas de las otras carpetas. A pesar de lo dicho, el valor obtenido en al accuracy del test determina que el modelo aprendió patrones relevantes que le permiten identificar con cierta claridad las opciones posibles.

La clase con mayor desempeño es sin duda la de las tijeras, pues si bien tiene una presición de 0.609, cuenta con un recall de 0.807, y un f1-score de 0.694, por lo que el modelo identifica correctamente la mayoría de las imágenes con tijeras.
La clase con mayor presición fue la piedra, pero a costa de un recall de 0.441 (el más bajo de las 3 opciones), y un f1-score de 0.550. Esto significa que el sistema falla muy pocas veces cuando detecta una piedra, pero que más de la mitad de las veces no detecta piedra cuando realmente lo es.
Los valores obtenidos en papel son bastante malos, pues con una presición de 0.492, recall de 0.540 y f1-score de 0.515, se determina que el modelo tiene complicaciones para distinguir cuando se trata de esta categoría.


### _Matriz de confusión se encuentra en el código_

La matriz de confusión obtenida muestra lo que se argumentó previamente. 

La clase mayormente reconocida fue la de las tijeras, y fue pocas veces confundida con piedra, aunque si 24 veces confundida con papel.

La clase piedra, tal como se indicó previamente cuenta con un recall de 0.441, lo que se puede observar en los resultados cuando la clase real era piedra, que solamente se determinó correctamente en 83 de todos los casos. De hecho, muchas veces el sistema confundió la clase piedra con papel, pues ocurrió en 74 casos.

Por último, en la clase papel se acertó en 95 de todos los casos, en donde el principal error fue que el sistema atribuyó 60 de los casos como tijeras. Este error puede haber ocurrido dada una posición de manos que tuviese patrones mas atribuibles a las características que el sistema determina como tijeras. 


### Visualización de aciertos

<img width="973" height="349" alt="Papel C o I" src="https://github.com/user-attachments/assets/2a6506ba-54e7-4351-a547-061db0c612a9" />
<img width="964" height="321" alt="Papel C O I 2" src="https://github.com/user-attachments/assets/08e0ab4d-ce1c-43ab-95e4-416cd16dc950" />
<img width="970" height="327" alt="Papel C O I 3" src="https://github.com/user-attachments/assets/c13ece21-7299-4532-a060-b112fd00d420" />

Las predicciones realizadas sobre el conjunto de prueba muestran que el modelo fue capaz de identificar correctamente la mayoría de los gestos. En las imágenes clasificadas correctamente se observa que el modelo reconoce adecuadamente las características distintivas de cada clase, incluso cuando existen pequeñas variaciones en la posición de la mano o en el fondo de la imagen.  

No obstante, también se identificaron algunos errores de clasificación. En particular, ciertas imágenes correspondientes a la clase paper fueron clasificadas como scissors. Esto ocurre principalmente cuando la mano aparece inclinada o con una perspectiva que modifica la forma característica del gesto, haciendo que algunos dedos sean interpretados como la forma de tijera.

Se pueden considerar ciertas causas las cuales produzcan este pequeño error de clasificación:  
* Variaciones en el ángulo de la mano.
* Diferencias de iluminación.
* Perspectivas que alteran la forma del gesto.

Estos resultados indican que el modelo aprendió correctamente los patrones principales, pero aún puede mejorar su capacidad de generalización frente a situaciones más complejas.

## Conclusión de los resultados  
En este proyecto, se implementó un sistema de clasificación de imágenes para reconocer los gestos de piedra, papel y tijeras mediante transferencia de aprendizaje. Se dió uso a el modelo preentrenado ResNet18, congelando su backbone y reemplazando la capa final por una nueva cabeza de clasificación adaptada a las 3 opciones presentes del dataset.

El entrenamiento mostró una evolución positiva, pues el accuracy de validación fue de 0.749, y si pérdida de validación disminuyó de forma sostenida. Esto indica que el modelo logró aprender características relevantes del dataset sin presentar overfitting considerable, pues se observó una diferencia moderada entre el rendimiento de entrenamiento y validación.

En la evaluación del testeo, se observó un accuracy del 0.5926, que si bien es un desempeño menor al accuracy de validación, el modelo definitivamente es mejor que un diseño azaroso, por lo que si existe un aprendizaje de los patrones existentes en las imágenes. La clase de scissors (tijeras) tuve el mejor resultado del modelo, con un recall de 0.807 y un f1-score de 0.694, mientras que la clases rock (piedra) y paper (papel) tuvieron un desempeño menor, lo que implica mayor dificultad de clasificación. 

La matriz de confusión logra confirmar estas dificultades descritas, pues el modelo confunde las clases reales de papel y piedra con prediciones de tijera y papel respectivamente.

La diferencia de resultados entre los accuracy de testeo y validación sugieren que el conjunto de testeo contiene variaciones que al modelo le resultan inusuales o distintos a los observados en el entrenamiento, lo que puede deberse a distintas posturas, distancias y posiciones de manos. A pesar de lo dicho, se demuestra que el modelo si aprendió cierta parte de la información como patrón, por lo que si es posible aplicar un clasificador funcional, aunque posiblemente se requiera de una base de datos más amplia y estandarizada, de modo que pueda mejorar la accuracy en términos generales.

Además de lo dicho, también hay que tener en cuenta que las limitaciones computacionales que se describieron previamente pueden haber influido en los resultados obtenidos, por lo que a trabajo futuro sería interesante documentar cómo se desempeña un modelo preentrenado más robusto y por supuesto con una base de datos de igual robustez.










