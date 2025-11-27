# Inferencia en tiempo real con TensorRT y OpenCV en Jetson Nano Orin

## 1. Introducción

**Contexto general de los procesadores de alto rendimiento**  
Los procesadores de alto rendimiento (High Performance Processors, HPP) constituyen el núcleo de los sistemas modernos de cómputo intensivo, diseñados para ejecutar grandes volúmenes de operaciones en paralelo y reducir drásticamente los tiempos de procesamiento. Su desarrollo surge ante la creciente demanda de inteligencia artificial, simulaciones científicas, análisis de datos y aplicaciones en tiempo real. Estos procesadores —como las GPU, TPU y FPGA— permiten maximizar la eficiencia energética y el rendimiento mediante arquitecturas especializadas, aprovechando el paralelismo masivo y la optimización del flujo de instrucciones. En conjunto, conforman la base tecnológica de los actuales sistemas de alto rendimiento (HPC).

**Motivación del uso de GPU en sistemas embebidos**  
Las GPU en sistemas embebidos, como la Jetson Nano Orin, ofrecen un equilibrio ideal entre potencia de cálculo y consumo energético. Su arquitectura masivamente paralela permite ejecutar redes neuronales convolucionales, algoritmos de visión por computador y procesamiento de video en tiempo real, tareas que serían ineficientes en CPU convencionales. Además, su integración con CUDA y TensorRT permite aprovechar los núcleos dedicados al cómputo vectorial, logrando acelerar significativamente la inferencia de modelos de inteligencia artificial en el borde (edge computing), reduciendo la dependencia de servidores externos y mejorando la autonomía del sistema embebido.

**Rol de TensorRT en la optimización de modelos de IA para inferencia en tiempo real**  
TensorRT es una plataforma de optimización e inferencia de alto rendimiento desarrollada por NVIDIA, orientada a desplegar modelos de IA de forma eficiente sobre GPU. A través de la conversión de modelos entrenados (ONNX) a motores optimizados (.plan), TensorRT aplica técnicas como fusión de capas, cuantización a FP16 o INT8 y eliminación de operaciones redundantes. Estas optimizaciones reducen la latencia y aumentan la velocidad de procesamiento sin sacrificar precisión. En entornos embebidos como Jetson Nano Orin, TensorRT permite alcanzar inferencias en tiempo real con recursos limitados, consolidándose como una herramienta esencial para la IA en el borde.

**Objetivo del repositorio y del presente documento**  
El presente repositorio tiene como objetivo demostrar la implementación práctica de un sistema de inferencia en tiempo real utilizando TensorRT y OpenCV en la Jetson Nano Orin. Este documento busca guiar al lector paso a paso en la configuración del entorno, conversión del modelo, ejecución del motor optimizado y análisis del rendimiento alcanzado. Se enfatiza la replicabilidad del proceso, la comprensión del flujo de inferencia y la medición de métricas clave como FPS, latencia y uso de GPU. El enfoque combina teoría y práctica, integrando los fundamentos de los procesadores de alto rendimiento con su aplicación en la aceleración de IA.


---

## 2. Arquitectura de Hardware

**Descripción de la Jetson Nano Orin**  
La Jetson Nano Orin es un dispositivo embebido desarrollado por NVIDIA para aplicaciones de inteligencia artificial en el borde (edge AI). Incorpora una arquitectura basada en Ampere, que combina eficiencia energética con gran capacidad de cómputo. Su diseño compacto permite ejecutar modelos de visión por computador, reconocimiento de objetos y redes neuronales profundas en tiempo real. Gracias a su integración con CUDA, TensorRT y DeepStream, se convierte en una plataforma ideal para el desarrollo de proyectos de inferencia acelerada. En este contexto, se utiliza como plataforma experimental para desplegar modelos optimizados con TensorRT y procesarlos mediante OpenCV.

**GPU integrada (núcleos CUDA, memoria, potencia)**  
La Jetson Nano Orin integra una GPU con arquitectura Ampere compuesta por 1024 núcleos CUDA y soporte nativo para operaciones de precisión mixta (FP16 e INT8). Posee configuraciones de memoria LPDDR5 de 8 GB o 16 GB, con un ancho de banda suficiente para manejar modelos de visión por computador en alta resolución. Su rendimiento alcanza hasta 40 TOPS, dependiendo de la configuración de potencia utilizada. Esta combinación de núcleos y memoria optimizada la convierte en un entorno ideal para la ejecución de inferencias en tiempo real, balanceando consumo energético, velocidad y capacidad de procesamiento simultáneo.

**Compatibilidad con TensorRT y versiones de JetPack**  
El dispositivo es totalmente compatible con JetPack 6.x, el entorno de desarrollo oficial de NVIDIA que integra los paquetes CUDA 12.6, cuDNN 9 y TensorRT 10.3. Esta combinación proporciona un ecosistema optimizado para el desarrollo y despliegue de modelos de inteligencia artificial en plataformas Jetson. TensorRT realiza la optimización del grafo de inferencia, aplicando técnicas como fusión de capas y cuantización, mientras que CUDA y cuDNN gestionan la aceleración de operaciones matriciales y convolucionales. Este soporte unificado entre hardware y software garantiza un flujo de trabajo eficiente, desde el entrenamiento del modelo hasta su ejecución con mínima latencia y máximo aprovechamiento de la GPU.

**Requisitos previos del sistema**  
Para la correcta implementación del repositorio `demoOpenCVRealtimeTensorRT`, se requiere un entorno con JetPack 6.x y los componentes CUDA Toolkit, cuDNN y TensorRT correctamente configurados. Es esencial contar con OpenCV y PyCUDA instalados, ya que gestionan la captura de video y la comunicación directa con la GPU. Además, se recomienda disponer de Python 3.10 y librerías científicas como NumPy para el preprocesamiento de imágenes. La verificación del entorno puede realizarse ejecutando comandos como `nvcc --version`, `trtexec` y `python -c "import tensorrt"`, asegurando que todos los módulos de aceleración estén activos y accesibles para la inferencia.
  
---

## 3. Configuración del Entorno

### 3.1 Instalación del sistema base  
La configuración del entorno en la **Jetson Nano Orin** comienza con la instalación de **JetPack 6.x (L4T 36.x)**, que incluye los componentes esenciales para el desarrollo en inteligencia artificial: **CUDA 12.6**, **cuDNN 9** y **TensorRT 10.3**.  
Se recomienda mantener el sistema actualizado ejecutando:  
```bash
sudo apt-get update && sudo apt-get upgrade

Luego, se debe configurar la conectividad de red y habilitar el acceso remoto mediante SSH. Este enfoque facilita la programación, depuración y despliegue sin necesidad de periféricos locales, optimizando el flujo de trabajo en entornos embebidos de desarrollo.

3.2 Instalación de dependencias

El repositorio demoOpenCVRealtimeTensorRT requiere instalar librerías clave para la ejecución de inferencias optimizadas con TensorRT y OpenCV.
Primero, habilite el entorno Python y las dependencias principales:

sudo apt-get update
sudo apt-get install python3-pip libopencv-dev
pip install numpy pycuda opencv-python tensorrt

Estas librerías garantizan la compatibilidad con CUDA y la correcta ejecución en GPU.

NumPy: operaciones matriciales y normalización de imágenes.
PyCUDA: transferencia eficiente entre CPU y GPU.
OpenCV: captura, preprocesamiento y visualización de video.
TensorRT: motor de inferencia optimizado para hardware NVIDIA.

Verifique la instalación ejecutando: python3 -c "import cv2, tensorrt; print('OpenCV:', cv2.__version__)"
Si no se reportan errores, el entorno está correctamente configurado.

## 4. Estructura del Repositorio

### Archivos principales  

El repositorio `demoOpenCVRealtimeTensorRT` presenta una estructura clara y modular, diseñada para facilitar la ejecución, optimización y comparación entre inferencia en **PyTorch** y **TensorRT** dentro de dispositivos Jetson.  

- **`torch_seg.py`** → Ejecuta la inferencia en tiempo real utilizando **PyTorch** y el modelo **DeepLabV3-ResNet50**, proporcionando un punto de referencia base antes de aplicar optimizaciones.  
- **`trt_seg.py`** → Implementa la inferencia acelerada mediante **TensorRT** y **PyCUDA**, aprovechando la GPU integrada de la **Jetson Nano Orin**. Gestiona la captura de video, la transferencia CPU↔GPU, la inferencia optimizada y la visualización en pantalla con métricas de rendimiento (FPS).  
- **`build_onnx.py`** → Exporta el modelo DeepLabV3 desde PyTorch a formato **ONNX**, etapa intermedia necesaria para generar el motor TensorRT (`.plan`).  
- **`requirements.txt`** → Lista las dependencias necesarias para garantizar la compatibilidad del entorno Python con CUDA, cuDNN, PyTorch y TensorRT.  

Esta organización permite al usuario ejecutar el pipeline completo: desde la exportación del modelo hasta su inferencia optimizada en hardware embebido.

---

### Flujo interno  

El flujo de ejecución del sistema sigue una secuencia bien definida:  

1. **Captura de video:**  
   Se inicia el flujo con `cv2.VideoCapture(0)`, obteniendo fotogramas en tiempo real desde una cámara USB o CSI conectada a la Jetson.  

2. **Preprocesamiento:**  
   Cada imagen capturada es redimensionada a **512×512 píxeles**, normalizada con los valores de **ImageNet (mean/std)** y transformada al formato tensorial NCHW.  

3. **Carga del modelo o motor:**  
   - En `torch_seg.py`, el modelo DeepLabV3-ResNet50 se carga desde PyTorch con pesos preentrenados en COCO/VOC.  
   - En `trt_seg.py`, se carga el motor **TensorRT (`model.plan`)** previamente generado, optimizado para la arquitectura **Ampere** de la Jetson Nano Orin.  

4. **Inferencia y postprocesamiento:**  
   La inferencia se ejecuta en GPU, y los resultados de segmentación se procesan para generar una **máscara coloreada** superpuesta al video original. Se muestran métricas de rendimiento en pantalla, como el **FPS promedio**, latencia y clases detectadas.  

Este flujo modular permite comparar fácilmente el rendimiento entre **PyTorch** y **TensorRT**, destacando la reducción de latencia y el aumento del throughput tras la optimización del modelo.


## 5. Conversión del Modelo a TensorRT

### Flujo de trabajo

1. **Obtener el modelo original**  
   Seleccionar un modelo entrenado (por ejemplo, DeepLabV3, YOLOv8 o ResNet) compatible con exportación a ONNX.

2. **Exportar a ONNX**  
   Convertir el modelo desde PyTorch utilizando `torch.onnx.export()` para generar el archivo intermedio `model.onnx`.

3. **Convertir a TensorRT**  
   Generar el motor optimizado ejecutando:  
   ```bash
   trtexec --onnx=model.onnx --saveEngine=model.plan --fp16

4. **Ajustar parámetros**
Modificar la precisión (FP16 o INT8), el tamaño de lote y la resolución de entrada según los recursos disponibles en la Jetson Nano Orin.

5. **Verificar el motor**
Confirmar la correcta creación del archivo model.plan con ls -lh model.plan antes de realizar la inferencia en tiempo real.



## 7. Benchmark y Análisis de Rendimiento

### Métricas a reportar  
El desempeño del sistema se mide a través del **FPS promedio**, la **latencia por inferencia** y el **uso de GPU**, monitoreado con la herramienta `tegrastats`. Estas métricas permiten evaluar la eficiencia del modelo optimizado con TensorRT sobre la Jetson Nano Orin, reflejando la capacidad del dispositivo para ejecutar inferencias en tiempo real con bajo consumo y alta estabilidad.

### Comparativas  
Se comparan los resultados obtenidos entre inferencia en **CPU y GPU**, destacando la mejora de rendimiento proporcionada por TensorRT. Las optimizaciones **FP16** e **INT8** reducen significativamente la latencia y elevan los FPS, manteniendo una precisión estable y demostrando el beneficio de la aceleración por hardware en entornos embebidos de IA.



## 8. Conclusiones

- La **Jetson Nano Orin** demuestra un equilibrio óptimo entre rendimiento computacional y eficiencia energética, posicionándose como una plataforma ideal para tareas de inferencia en el borde.  
- El uso de **TensorRT** mejora notablemente la velocidad y la latencia frente a la inferencia directa con PyTorch, aprovechando de forma eficiente los núcleos CUDA y las optimizaciones FP16/INT8.  
- Las principales limitaciones detectadas se asocian al **uso de memoria** y a la **resolución máxima de entrada**, que pueden afectar la estabilidad del rendimiento.  
- Como proyección, se plantea la integración con **NVIDIA DeepStream** y la ejecución en modo **batch inference** para optimizar aún más el procesamiento en tiempo real.
