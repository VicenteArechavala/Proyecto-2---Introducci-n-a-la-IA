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
Elección del modelo: ResNet18  

Para este proyecto se seleccionó ResNet18 como arquitectura principal de Deep Learning. El objetivo es clasificar imágenes correspondientes a los gestos de piedra, papel y tijera, un problema de clasificación multiclase donde las diferencias entre categorías dependen de características visuales como la forma de la mano, la posición de los dedos y el contorno del gesto. ResNet18 ya posee la capacidad de reconocer patrones visuales generales, como bordes, líneas, texturas y formas. Mediante Transfer Learning, este conocimiento previo puede adaptarse al reconocimiento de los tres gestos del dataset, reduciendo significativamente el tiempo de entrenamiento y la cantidad de datos necesarios para obtener un buen desempeño.

* Ventajas: Presenta un buen equilibrio entre precisión y costo computacional siendo mas liviana que otros modelos, es una arquitectura ampliamente utilizada y validada en tareas de clasificación de imágenes, lo que garantiza estabilidad y disponibilidad de implementaciones optimizadas y posee una buena capacidad de generalización cuando se combina con técnicas de regularización y aumento de datos.
  
* Limitaciones: Aunque ResNet18 ofrece un buen rendimiento, su capacidad de representación es menor que la de arquitecturas más profundas, por lo que podría presentar dificultades frente a problemas con una gran cantidad de clases o patrones extremadamente complejos. Su desempeño depende de que las imágenes utilizadas, se espera que representen adecuadamente las variaciones presentes en un escenario real, como cambios de iluminación, posición o fondo.
  
* Pertinencia: La elección de ResNet18 es pertinente porque el problema consiste en reconocer patrones visuales bien definidos dentro de imágenes y clasificarlos en una de tres categorías. El modelo ofrece una combinación adecuada entre precisión, velocidad de entrenamiento y consumo de recursos computacionales, lo que lo convierte en una alternativa apropiada.


