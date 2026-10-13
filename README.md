# Informe Técnico y Comparación Experimental: CNN vs MLP en Sign Language MNIST

**Institución**: Duoc UC — Escuela de Informática y Telecomunicaciones  
**Asignatura**: Técnicas Avanzadas de Machine Learning I (Sigla: TLY1102 / TLY1101)  
**Evaluación**: Evaluación Parcial N°2 — Clasificación de Imágenes con Redes Neuronales Convolucionales (CNN)  
**Notebook Principal**: [`notebooks/Sign_Language_MNIST_CNN.ipynb`](file:///c:/Users/Eduardo/Desktop/EV2_TAML/notebooks/Sign_Language_MNIST_CNN.ipynb) *(o en la raíz del repositorio)*  
**Framework**: TensorFlow 2.x / Keras (sin capas de regularización, alineado estrictamente con la rúbrica)  
**Variante Específica del Equipo**: 10 clases del alfabeto dactilológico ASL (`A, B, C, D, E, I, L, M, N, O`) con semilla fija `seed = 42069`.

---

## 1. Matriz de Responsabilidades

En cumplimiento con las exigencias formales de la rúbrica docente (prevención de penalización del 10%):

| Integrante del Equipo | Rol Principal | Tareas y Responsabilidades Desarrolladas | Firma / Estado |
| :--- | :--- | :--- | :---: |
| **Integrante 1 (Eduardo)** | Líder Técnico & Modelado CNN | Preparación y preservación del formato matricial 2D, diseño e implementación de la arquitectura CNN en Keras, entrenamiento controlado sin capas de regularización. | Conforme |
| **Integrante 2** | Análisis Experimental & Métricas | Evaluación comparativa (Accuracy, Precision, Recall, F1 macro), generación de matrices de confusión y tabla de desempeño experimental MLP vs CNN. | Conforme |
| **Integrante 3** | Interpretabilidad & Explicabilidad | Extracción e interpretación de Mapas de Características (`conv_1`), implementación de Grad-CAM por clase y auditoría de fallas morfológicas. | Conforme |

---

## 2. Descripción del Problema y Justificación de la Variante

### 2.1 Contexto del Problema
La comunicación dactilológica estática del alfabeto ASL presenta retos de visión artificial debido a variaciones morfológicas complejas en baja resolución ($28 \times 28$ píxeles). En la Evaluación 1 se implementó un Perceptrón Multicapa (MLP) que requirió aplanar la imagen ($784$ vectores unidimensionales). En esta Evaluación 2 se desarrolla una **Red Neuronal Convolucional (CNN)** para explotar la correlación espacial 2D y la extracción jerárquica de características visuales.

### 2.2 Variante Conservada (Criterio Anti-Copia)
Se conserva rigurosamente la misma variante de la Evaluación 1:
$$\text{Variante} = \{\text{'A', 'B', 'C', 'D', 'E', 'I', 'L', 'M', 'N', 'O'}\}$$
* **Etiquetas originales**: `[0, 1, 2, 3, 4, 8, 11, 12, 13, 14]`
* **Re-mapeo continuo**: `{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 8: 5, 11: 6, 12: 7, 13: 8, 14: 9}`
* **Semilla determinista**: `seed = 42069`
* **Partición de datos**:
  * **Entrenamiento (Train)**: $8,990$ muestras ($80\%$ del conjunto filtrado).
  * **Validación (Val)**: $2,248$ muestras ($20\%$ estratificado).
  * **Prueba (Test Set)**: $3,244$ imágenes no vistas nunca durante el ajuste de pesos.

---

## 3. Preparación de Datos: Diferencia entre CNN y MLP

1. **Conservación de la Topología 2D**:
   * **MLP**: Requiere vectorizar la imagen a $1\text{D}$ ($28 \times 28 \to 784$), perdiendo las relaciones de contigüidad espacial entre píxeles adyacentes verticales y diagonales.
   * **CNN**: Mantiene la estructura tensorial espacial `(28, 28, 1)` mediante `reshape(-1, 28, 28, 1)`, preservando la matriz bidimensional de cada seña.
2. **Normalización Min-Max**: Escalamiento de intensidades $[0, 255] \to [0.0, 1.0]$ dividiendo por `255.0` para estabilizar el cálculo del descenso de gradiente.

---

## 4. Diseño y Justificación de la Arquitectura Convolucional

Para dar estricto cumplimiento a la pauta y evitar la **penalización del 20% por uso de capas extra (capas de regularización)**, la CNN no utiliza `Dropout`, `BatchNormalization` ni penalizadores $L_1/L_2$.

### 4.1 Arquitectura Implementada

```
Entrada: Imagen (28, 28, 1)
   │
   ▼
[ Conv2D: 32 filtros, kernel 3x3, activación ReLU ]  ──> Salida: (26, 26, 32)  | Params: 320
   │
   ▼
[ MaxPooling2D: ventana 2x2, stride 2 ]              ──> Salida: (13, 13, 32)  | Params: 0
   │
   ▼
[ Flatten ]                                          ──> Salida: (5408,)       | Params: 0
   │
   ▼
[ Dense: 128 neuronas, activación ReLU ]             ──> Salida: (128,)        | Params: 692,224
   │
   ▼
[ Dense (Salida): 10 neuronas, activación Softmax ]  ──> Salida: (10,)         | Params: 1,290
```

### 4.2 Justificación Técnica de Componentes
* **Filtros Convolucionales $3\times 3$ (32 kernels)**: Tamaño óptimo para capturar características morfológicas locales de baja escala (bordes de los dedos, curvaturas de la palma y contrastes de falanges) con una cantidad mínima de parámetros ($3\times 3 \times 1 \times 32 + 32 = 320$).
* **Función de Activación ReLU**: Introduce no linealidad computacionalmente eficiente y mitiga el desvanecimiento del gradiente (*vanishing gradient*).
* **MaxPooling2D ($2\times 2$)**: Reduce la dimensionalidad espacial a la mitad ($26\times 26 \to 13\times 13$), otorgando invarianza frente a pequeñas traslaciones y reduciendo el costo de memoria.
* **Capa Densa (128 unidades)**: Integra las combinaciones globales de las activaciones convolucionales extraídas antes de la clasificación final.
* **Capa de Salida Softmax (10 unidades)**: Distribución de probabilidades normalizada sobre las 10 letras del alfabeto.
* **Optimizador e Hiperparámetros**: Adam con tasa de aprendizaje adaptativa ($lr = 0.001$), función de pérdida `sparse_categorical_crossentropy`, 10 épocas y batch size de 64.

---

## 5. Comparación Experimental Rigurosa: CNN vs MLP

Ambos modelos fueron entrenados y evaluados bajo un **protocolo de control idéntico** (mismo split, 10 épocas, batch size 64, Adam $lr=0.001$, evaluados sobre las $3,244$ imágenes de prueba):

| Dimensión de Comparación | MLP Control (Evaluación 1) | Red Neuronal Convolucional (CNN) | Diferencia / Impacto |
| :--- | :---: | :---: | :--- |
| **Exactitud en Test (Accuracy)** | 87.58% | **92.66%** | **+5.08% a favor de CNN** 🏆 |
| **Precision Macro** | 86.79% | **92.43%** | **+5.64% a favor de CNN** |
| **Recall Macro** | 87.23% | **91.84%** | **+4.61% a favor de CNN** |
| **F1-Score Macro** | 0.8651 | **0.9177** | **+0.0526 a favor de CNN** |
| **Imágenes Falladas (de 3,244)** | 403 errores | **238 errores** | **165 errores menos** (-40.9% error) |
| **Parámetros Totales** | **101,770** | 693,962 | CNN tiene ~6.8x más parámetros |
| **Tiempo de Entrenamiento (10 épocas)**| **7.23 s** | 54.72 s | CNN requiere mayor cómputo |
| **Latencia de Inferencia (ms/imagen)**| **0.218 ms** | 0.336 ms | MLP es marginalmente más rápido |

> **Análisis de la Paradoja Paramétrica**:  
> Aunque las capas convolucionales son reconocidas por compartir pesos (`Conv2D` aporta apenas 320 parámetros), la arquitectura CNN evaluada suma **693,962 parámetros** debido a que la salida tras un solo pooling de $13 \times 13 \times 32$ canales produce un vector aplanado de **5,408 entradas**, conectándose a la capa densa de 128 neuronas ($5,408 \times 128 + 128 = 692,224$ parámetros). Esta capa densa concentra el **99.7% del peso del modelo**.

---

## 6. Evaluación de Desempeño y Diagnóstico por Clase

### 6.1 Reporte por Letra en el Test Set (CNN)
* **Clases con Desempeño Casi Perfecto ($F_1 \ge 0.97$)**:
  * **B** ($F_1 = 1.00$): Reconocimiento infalible de los 4 dedos extendidos verticalmente.
  * **C** ($F_1 = 1.00$): Silueta cóncava en arco sumamente distintiva.
  * **L** ($F_1 = 1.00$): Ángulo recto característico entre pulgar e índice.
  * **E** ($F_1 = 0.98$): Puño semicerrado con dedos retraídos.
  * **I** ($F_1 = 0.97$): Excelente detección del meñique vertical aislado.
  * **D** ($F_1 = 0.95$): Precisión muy alta en índice vertical.
* **Clase Crítica / Cuello de Botella Morfológico**:
  * **N** (Recall: 55%, Precision: 72%, $F_1 = 0.62$): Representa el desafío más complejo. Al consistir en un puño cerrado donde el pulgar cruza bajo dos dedos, a resolución $28 \times 28$ su contraste es casi indistinguible de la letra `A` (pulgar lateral) y `M` (pulgar bajo tres dedos).

---

## 7. Interpretabilidad y Visualización

### 7.1 Mapas de Características (Feature Maps)
Al propagar una muestra de prueba a través de `conv_1`, los 32 filtros generan mapas de $26 \times 26$ donde se constata que:
1. **Filtros iniciales** actúan como detectores de bordes horizontales y verticales.
2. Otros filtros aíslan gradientes de iluminación en el contorno de la mano y las puntas de los dedos, demostrando que la red no memoriza píxeles aislados sino patrones morfológicos contiguos.

### 7.2 Grad-CAM por Clase (Activation Maps)
La técnica de *Gradient-weighted Class Activation Mapping* confirma visualmente sobre las 10 clases que:
* Para letras con dedos extendidos (`B`, `D`, `I`, `L`), las áreas de máxima activación (zonas cálidas en rojo/amarillo) se ubican exactamente sobre la punta y extensión del dedo correspondiente.
* Para gestos circulares (`C`, `O`), la atención de la red se concentra en el arco exterior y la apertura de la mano, descartando el fondo neutro.

---

## 8. Conclusiones y Comparativa Teórica

1. **Capacidad de Generalización**: La CNN supera ampliamente al MLP en el conjunto de prueba independiente ($92.66\%$ vs $87.58\%$), reduciendo los errores de clasificación en más de un $40\%$.
2. **Invarianza Espacial**: La presencia de convolución y pooling permite tolerar pequeñas traslaciones y variaciones anatómicas locales que el MLP confunde al tratar las entradas como píxeles estáticos e independientes.
3. **Compromiso Cómputo-Rendimiento (Trade-off)**: La CNN ofrece mayor precisión y robustez pero requiere un tiempo de entrenamiento superior (~54 s vs ~7 s) y una huella de memoria mayor en la capa de transición densa.

---

## 9. Estructura del Repositorio Recomendada

Para cumplir con el estándar profesional solicitado en el **Detalle de la Entrega** (`data/`, `notebooks/`, `models/`, `images/`, `outputs/` y `README.md`):

```
EV2_TAML/
├── data/                                 # Conjunto de datos oficial
│   ├── sign_mnist_train.csv              # 27,455 muestras de entrenamiento originales
│   └── sign_mnist_test.csv               # 7,172 muestras de prueba originales
├── notebooks/                            # Notebooks reproducibles
│   ├── Sign_Language_MNIST_CNN.ipynb     # Notebook principal Evaluacion 2 (CNN vs MLP)
│   └── Sign_Language_MNIST_MLP.ipynb     # Notebook referencial Evaluacion 1 (MLP)
├── outputs/                              # Graficos y artefactos exportados por el codigo
│   ├── curvas_comparacion_mlp_cnn.png    # Curvas de perdida y exactitud por epoca
│   ├── matrices_confusion_mlp_cnn.png    # Matrices de confusion comparadas en test
│   ├── comparacion_mlp_cnn.csv           # Tabla de metricas cuantitativas
│   ├── ejemplos_mlp_cnn.png              # Galeria de aciertos y errores cualitativos
│   ├── feature_maps_conv1.png            # Visualizacion de mapas de caracteristicas
│   └── cnn_gradcam_por_clase.png         # Mapas de activacion Grad-CAM por letra
├── images/                               # Imagenes ilustrativas de apoyo y documentacion
│   ├── amer_sign2.png
│   ├── amer_sign3.png
│   └── american_sign_language.PNG
├── models/                               # Modelos entrenados serializados (opcional)
├── guion_presentacion_10min.md           # Guion cronometrado para defensa oral
├── TLY1102-E2.pdf                        # Pauta oficial del encargo
├── TLY1102-E2-Rubrica.pdf                # Rubrica de evaluacion docente
└── README.md                             # Informe tecnico oficial (este documento)
```

---

## 10. Instrucciones de Reproducibilidad

1. Clonar el repositorio y posicionarse en la carpeta raíz:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd EV2_TAML
   ```
2. Instalar dependencias requeridas:
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn
   ```
3. Ejecutar el notebook secuencialmente (`Restart Kernel and Run All Cells`) en Jupyter Lab o VS Code:
   ```bash
   jupyter lab Sign_Language_MNIST_CNN.ipynb
   ```
   *(Todos los gráficos, matrices y reportes se guardarán automáticamente en la carpeta `outputs/`)*.
