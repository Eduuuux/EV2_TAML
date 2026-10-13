# Guión de Presentación y Defensa Técnica (10 Minutos)
## Evaluación Parcial N°1 — Técnicas Avanzadas de Machine Learning I (TLY1101 / TLY1102)
### Proyecto: Clasificación de Lenguaje de Señas (Sign Language MNIST) con Redes Neuronales MLP (TensorFlow / Keras)

**Institución**: Duoc UC  
**Modalidad**: Presentación de evidencias y defensa técnica del proyecto (10 minutos estrictos + preguntas cruzadas individuales)  
**Estructura**: 7 Diapositivas clave diseñadas para cumplir al 100% con los indicadores de la rúbrica (**IE1 a IE6**).

---

### Diapositiva 1: Problema de Negocio, Objetivos y KPIs (Tiempo: 1.5 min)
**Ponente 1**:
- **Problema de Negocio**: *"Buenas tardes, profesor. Presentamos nuestro proyecto de tecnología asistiva e inclusión digital para la comunidad sorda o con discapacidad auditiva. El reto es traducir automáticamente señas estáticas del alfabeto dactilológico en tiempo real para aplicaciones de atención pública y educación inclusiva."*
- **Objetivos**: *"Implementar un flujo integral de aprendizaje supervisado bajo la metodología CRISP-DM, utilizando redes neuronales totalmente conectadas (MLP) sobre el dataset Sign Language MNIST, evaluando sus capacidades y límites en visión artificial."*
- **Variante del Equipo**: *"Para garantizar un trabajo único e infalsificable, seleccionamos una variante de 10 clases dactilológicas (5 vocales y 5 consonantes clave: `A, B, C, D, E, I, L, M, N, O`) particionadas con la semilla fija `seed=42069`."*
- **KPIs de Negocio**:
  1. **Accuracy Global ($\ge 85\%$)**: Logrado **89.77%** en prueba (2,912 de 3,244).
  2. **F1-Score Macro ($\ge 0.80$)**: Logrado **0.8854**.
  3. **Latencia de Inferencia ($\le 10$ ms)**: Logrado **$0.068$ ms / imagen** (óptimo para tiempo real).
  4. **Calidad de Datos**: **0 nulos (100% de integridad)**.

---

### Diapositiva 2: Análisis Exploratorio de Datos (EDA) y Calidad (Tiempo: 1.5 min)
**Ponente 1 / Ponente 2**:
- **Estructura del Dataset**: *"Trabajamos con imágenes en escala de grises de $28 \times 28$ píxeles, representadas por 784 variables numéricas entre 0 y 255."*
- **Identificación de Patrones**:
  - *Bordes y Contraste*: Marcada diferencia lumínica entre la palma y el fondo.
  - *Proyecciones Dactilares*: Señas con dedos extendidos ('L', 'B', 'D') presentan activaciones localizadas en sectores superiores, mientras que puños cerrados ('A', 'E', 'M', 'N') concentran la masa de píxeles en el centro.
- **Gráficos EDA Generados**:
  - *Distribución de Clases*: Confirmamos un dataset perfectamente balanceado (~1,100 muestras por clase en train).
  - *Histograma de Intensidad*: Revela una distribución bimodal de brillo que justifica la normalización.

---

### Diapositiva 3: Preparación y Preprocesamiento de Datos (Tiempo: 1.5 min)
**Ponente 2**:
- **Transformaciones Clave**:
  1. **Normalización Min-Max**: *"Escalamos los valores de píxel de $[0, 255]$ al rango $[0.0, 1.0]$ dividiendo por 255, asegurando estabilidad numérica en el gradiente."*
  2. **Vectorización / Aplanamiento**: *"Aplanamos la matriz $28 \times 28$ a un vector unidimensional $x \in \mathbb{R}^{784}$ para alimentar la capa de entrada del MLP."*
  3. **Partición de Datos Estratificada**: *"Separamos $8,990$ muestras para entrenamiento (80%), $2,248$ para validación (20%) y reservamos un conjunto de prueba de $3,244$ muestras no vistas jamás durante el entrenamiento."*

---

### Diapositiva 4: Diseño Arquitectónico con TensorFlow / Keras (Tiempo: 1.5 min)
**Ponente 2 / Ponente 3**:
- **Alineación con TLY1102**: *"Siguiendo las buenas prácticas vistas en la sesión 1.4 de la asignatura, diseñamos la función modular `build_classifier` en Keras."*
- **Decisiones Técnicas Fundamentadas**:
  - *Capa de Entrada*: `Input(shape=(784,))`.
  - *Capas Ocultas*: Activación **ReLU** para evitar el desvanecimiento del gradiente.
  - *Capa de Salida*: **Softmax** de 10 neuronas (probabilidades normalizadas que suman 1.0).
  - *Función de Pérdida*: `sparse_categorical_crossentropy`.
  - *Optimizador*: `Adam(learning_rate=0.001)`.
- **Comparación Experimental**:
  - *Pequeña (128)*: 1 capa oculta de 128 neuronas.
  - *Mediana (256, 128)*: 2 capas ocultas.
  - *Grande (512, 256, 128 con Dropout)*: 3 capas con Dropout al 20%.

---

### Diapositiva 5: Entrenamiento y Curvas de Aprendizaje (Tiempo: 1.5 min)
**Ponente 3**:
- **Monitoreo y Regularización**: *"Implementamos `EarlyStopping` monitoreando `val_loss` con paciencia de 8 épocas y restaurando los mejores pesos."*
- **Resultados de Validación**:
  - Tanto la Pequeña como la Mediana alcanzaron el **100% de exactitud en validación** (99.73% en la Grande, detenida en la época 22).
  - La arquitectura **Pequeña (128)** fue seleccionada como óptima por su máxima eficiencia computacional ($13.39$ segundos vs $22.99$ segundos de la Grande) y menor complejidad paramétrica.
- **Curvas de Aprendizaje (`plot_history`)**: *"Registramos la evolución de Loss y Accuracy lado a lado, verificando convergencia suave sin divergencia entre entrenamiento y validación."*

---

### Diapositiva 6: Evaluación en Test Set y Auditoría de Resultados (Tiempo: 1.5 min)
**Ponente 3**:
- **Resultados Globales en Test (3,244 imágenes)**:
  - **Accuracy**: **89.77%** (2,912 aciertos de 3,244).
  - **Precision Macro**: **88.90%** | **Recall Macro**: **88.81%** | **F1-Score**: **0.8854** | **Latencia**: **0.068 ms**.
- **Matriz de Confusión (`ConfusionMatrixDisplay`)**:
  - Rendimiento perfecto o sobresaliente en letras con silueta clara: **'C' (100% Precision, 100% Recall y 1.00 F1)**, **'L' (100% Recall)**, **'A' y 'E' (100% Recall)**, **'B' (97% Precision)**.
- **Auditoría de Aciertos y Errores**:
  - *Aciertos*: Confianza superior al $99.9\%$ en señas con contorno geométrico distintivo.
  - *Errores Principales*: Confusiones focalizadas entre **'N' y 'A' (81 casos)** y entre **'O' y 'A' (48 casos)**. Ambas señas corresponden a puños cerrados donde la posición sutil del dedo pulgar bajo los nudillos se difumina al aplanar la imagen en $28 \times 28$.

---

### Diapositiva 7: Limitaciones de MLP, Impacto Ético y Conclusión (Tiempo: 1.0 min)
**Todos / Conclusión Grupal**:
- **Limitaciones Demostradas del MLP**:
  1. *Destrucción de la estructura espacial 2D* al vectorizar las imágenes ($28 \times 28 \to 784$).
  2. *Falta de invarianza a la traslación* (si la mano se desplaza 3 píxeles, la red falla).
  3. *Explosión paramétrica* en imágenes de alta resolución ($>77$ millones de pesos por capa en imágenes HD).
- **Impacto Ético**: *"Es indispensable auditar sesgos por tono de piel, anatomía manual e iluminación para no crear barreras de exclusión en aplicaciones asistivas."*
- **Conclusión**: *"Cumplimos al 100% con los KPIs fijados, demostrando un dominio completo del flujo de deep learning supervisado en Keras. ¡Muchas gracias!"*

---

### Preguntas Cruzadas Típicas del Docente y Cómo Responderlas:

1. **¿Por qué usaron `sparse_categorical_crossentropy` en lugar de `categorical_crossentropy`?**
   * *Respuesta*: Porque nuestras etiquetas son enteros del $0$ al $9$. `sparse_categorical_crossentropy` permite calcular la pérdida directamente sobre enteros sin necesidad de transformar las etiquetas a vectores One-Hot encoding, ahorrando memoria y cómputo.

2. **¿Por qué la red confunde más la letra 'N' con la 'A' o la 'M'?**
   * *Respuesta*: Porque en ASL todas son señas de puño cerrado. La única diferencia física es dónde se coloca el pulgar respecto a los otros dedos. Al aplanar la imagen de $28 \times 28$ a un vector de 784 píxeles, la red pierde la relación espacial 2D entre dedos adyacentes y el detalle del pulgar se difumina.

3. **¿Para qué sirve `EarlyStopping` con `restore_best_weights=True`?**
   * *Respuesta*: Monitorea la pérdida en el conjunto de validación (`val_loss`). Si la red empieza a sobreajustarse y la pérdida de validación deja de mejorar durante 8 épocas consecutivas, detiene el entrenamiento automáticamente y restablece los pesos de la época con menor error.

4. **¿Cómo resolverían la falta de invarianza a la traslación en la siguiente entrega?**
   * *Respuesta*: Transicionando a una **Red Neuronal Convolucional (CNN)**, la cual utiliza filtros convolucionales con pesos compartidos y capas de max-pooling capaces de detectar bordes y características sin importar en qué posición del encuadre se encuentre la mano.
