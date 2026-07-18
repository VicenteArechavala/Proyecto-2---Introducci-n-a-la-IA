# Proyecto-2---Introduccion-a-la-IA
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
El proyecto utilizará el dataset Rock, Paper, Scissors Dataset. Contiene imágenes de gestos de manos correspondientes al juego piedra, papel o tijera, capturadas mediante una cámara web.
* Rock: Mano cerrada formando una piedra.
* Paper: Mano abierta representando papel.
* Scissors: Dedo índice y medio extendidos formando una tijera.

**Justificación del modelo**  
ResNet18  

Para este proyecto se seleccionó ResNet18 como arquitectura principal de Deep Learning. El objetivo es clasificar imágenes correspondientes a los gestos de piedra, papel y tijera, un problema de clasificación multiclase donde las diferencias entre categorías dependen de características visuales como la forma de la mano, la posición de los dedos y el contorno del gesto. ResNet18 ya posee la capacidad de reconocer patrones visuales generales, como bordes, líneas, texturas y formas. Mediante Transfer Learning, este conocimiento previo puede adaptarse al reconocimiento de los tres gestos del dataset, reduciendo significativamente el tiempo de entrenamiento y la cantidad de datos necesarios para obtener un buen desempeño.

* Ventajas: Presenta un buen equilibrio entre precisión y costo computacional siendo mas liviana que otros modelos, es una arquitectura ampliamente utilizada y validada en tareas de clasificación de imágenes, lo que garantiza estabilidad y disponibilidad de implementaciones optimizadas y posee una buena capacidad de generalización cuando se combina con técnicas de regularización y aumento de datos.
  
* Limitaciones: Aunque ResNet18 ofrece un buen rendimiento, su capacidad de representación es menor que la de arquitecturas más profundas, por lo que podría presentar dificultades frente a problemas con una gran cantidad de clases o patrones extremadamente complejos. Su desempeño depende de que las imágenes utilizadas, se espera que representen adecuadamente las variaciones presentes en un escenario real, como cambios de iluminación, posición o fondo.
  
* Pertinencia: La elección de ResNet18 es pertinente porque el problema consiste en reconocer patrones visuales bien definidos dentro de imágenes y clasificarlos en una de tres categorías. El modelo ofrece una combinación adecuada entre precisión, velocidad de entrenamiento y consumo de recursos computacionales, lo que lo convierte en una alternativa apropiada.

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

**Resultados**  
Los resultados obtenidos muestran que el modelo ResNet18, entrenado con Crossentropy y optimizado mediante AdamW, logró aprender correctamente las características principales de los tres gestos del dataset. A partir de las imágenes y gráficos generados, se puede notar.  

<img width="593" height="398" alt="Cantidad de papel, piedra y tijeras" src="https://github.com/user-attachments/assets/5b876159-e443-4ab2-95cc-3344c248d615" />   

Antes del entrenamiento se analizó la cantidad de imágenes por clase. La distribución mostró que las tres categorías (paper, rock y scissors) poseen una cantidad de ejemplos muy similar, con aproximadamente 330 a 350 imágenes por clase. Esto indica que el dataset se encuentra balanceado, disminuyendo el riesgo de que el modelo favorezca una categoría por sobre las demás durante el entrenamiento.  

<img width="874" height="294" alt="Piedra" src="https://github.com/user-attachments/assets/4114a51b-feab-49c3-aecc-f7f6fc2feffd" />
<img width="874" height="284" alt="Papel" src="https://github.com/user-attachments/assets/f8b6bb5b-9516-4b56-8f29-e764d8b1d76d" /> 
<img width="874" height="290" alt="Tijeras" src="https://github.com/user-attachments/assets/2bbb1af6-b3c3-4b5c-95e9-d5f753637cfd" />  

Como se puede notar, hay tres imagenes para cada una realizadas de forma aleatoria. El modelo fue capaz de poder definir la posicion de la mano si era piedra, papel o tijera, Esto es bastante importante ya que luego de poder definirlo, comienza el trabajo del modelo y metricas ocupadas para empezar a ver si lo define de forma correcta o incorrecta. 






