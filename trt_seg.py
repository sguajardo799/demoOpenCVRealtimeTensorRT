import time
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# =========================
# Constantes
# =========================
IM_SIZE = 512
MODEL_PLAN = "model.plan"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def make_palette(n: int = 256, seed: int = 0) -> np.ndarray:
    """
    Genera una paleta de colores indexada (n x 3, uint8) para colorear máscaras.

    Args:
        n: número de clases/colores.
        seed: semilla para reproducibilidad.

    Returns:
        Paleta RGB de tamaño (n, 3). Se fuerza color negro en el índice 0 (background).
    """
    rng = np.random.RandomState(seed)
    pal = (rng.rand(n, 3) * 255).astype(np.uint8)
    pal[0] = np.array([0, 0, 0], np.uint8)  # clase 0: fondo
    return pal


PALETTE = make_palette(256)

# Etiquetas VOC (DeepLabV3 torchvision preentrenado).
categories = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def colorize_mask(mask_uint8: np.ndarray) -> np.ndarray:
    """
    Convierte una máscara de clases (H, W) en una imagen RGB (H, W, 3)
    usando la paleta global PALETTE.

    Args:
        mask_uint8: máscara de clases uint8 (H, W).

    Returns:
        Imagen RGB coloreada (H, W, 3) uint8.
    """
    return PALETTE[mask_uint8]


def overlay_mask(bgr: np.ndarray, mask_color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Superpone la máscara coloreada sobre el frame BGR original.

    Args:
        bgr: frame original (H, W, 3) en BGR.
        mask_color: máscara coloreada RGB (h, w, 3) sin garantizar mismo tamaño.
        alpha: opacidad de la máscara superpuesta.

    Returns:
        Imagen BGR (H, W, 3) con la máscara superpuesta.
    """
    mask_resized = cv2.resize(mask_color, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(bgr, 1.0 - alpha, mask_resized, alpha, 0.0)


# =========================
# Preprocesado CPU
# =========================
def preprocess_bgr_cpu(frame_bgr: np.ndarray, size: int = IM_SIZE) -> np.ndarray:
    """
    Preprocesa un frame BGR para el modelo: BGR→RGB, resize, normaliza ImageNet,
    reordena a NCHW y devuelve un tensor contiguo float32 (1, 3, size, size).

    Notas de memoria:
        - Se crea un array destino con np.empty(...) y luego se copia la vista CHW.
          Esto garantiza contigüidad C (requisito para memcpy_htod).

    Args:
        frame_bgr: frame BGR uint8 (H, W, 3).
        size: tamaño cuadrado de entrada del modelo.

    Returns:
        x: tensor float32 contiguo shape (1, 3, size, size).
    """
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std  = np.asarray(IMAGENET_STD,  dtype=np.float32)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1))              # (3, H, W), puede ser no contiguo
    x = np.empty((1, 3, size, size), np.float32)    # buffer contiguo
    x[0] = chw                                      # copia -> contiguo
    return x


# =========================
# TensorRT init
# =========================
def load_engine_and_alloc(plan_path: str) -> dict:
    """
    Carga el engine TensorRT, crea el contexto de ejecución y reserva
    buffers en device para entrada/salida + una stream CUDA.

    Suposiciones:
        - Entrada fija (1, 3, IM_SIZE, IM_SIZE) float32.
        - La salida se lee del engine/contexto (shape y dtype).

    Args:
        plan_path: ruta al archivo .plan (engine serializado).

    Returns:
        Dict con objetos clave para inferencia:
            engine   : trt.ICudaEngine (metadatos, red optimizada)
            ctx      : trt.IExecutionContext (estado de ejecución)
            inp_name : str (nombre tensor de entrada)
            out_name : str (nombre tensor de salida)
            out_shape: tuple[int,...] (shape de salida)
            out_dtype: np.dtype (dtype de salida)
            d_in     : pycuda.driver.DeviceAllocation (buffer input en VRAM)
            d_out    : pycuda.driver.DeviceAllocation (buffer output en VRAM)
            stream   : pycuda.driver.Stream (cola CUDA para copias/ejecución)
    """
    logger = trt.Logger(trt.Logger.WARNING)
    with open(plan_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    # Descubrir nombres I/O (API de tensores). Evita hardcodear índices de binding.
    inp_name = next(n for n in engine if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_name = next(n for n in engine if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

    # Entrada fija → bytes conocidos
    bytes_in = 1 * 3 * IM_SIZE * IM_SIZE * np.dtype(np.float32).itemsize
    d_in = cuda.mem_alloc(bytes_in)  # buffer de entrada en device (vida larga)

    # Salida: leemos shape y dtype del engine/contexto
    out_shape = tuple(int(engine.get_tensor_shape(out_name)[i]) for i in range(len(engine.get_tensor_shape(out_name))))
    out_dtype = trt.nptype(engine.get_tensor_dtype(out_name))
    bytes_out = int(np.prod(out_shape)) * np.dtype(out_dtype).itemsize
    d_out = cuda.mem_alloc(bytes_out)  # buffer de salida en device (vida larga)

    stream = cuda.Stream()  # orden FIFO + asincronía

    return {
        "engine": engine,
        "ctx": ctx,
        "inp_name": inp_name,
        "out_name": out_name,
        "out_shape": out_shape,
        "out_dtype": out_dtype,
        "d_in": d_in,
        "d_out": d_out,
        "stream": stream
    }


# =========================
# Main (video)
# =========================
if __name__ == "__main__":
    # --- CPU: limitar threads de OpenCV para evitar overhead al preprocesar ligero ---
    cv2.setNumThreads(1)

    # --- Cargar engine y reservar recursos de GPU una sola vez ---
    st = load_engine_and_alloc(MODEL_PLAN)

    # --- Abrir cámara ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    print("Presiona 'q' para salir.")
    t0 = time.perf_counter()
    n_frames = 0
    window_name = "TRT Seg (simple)"

    while True:
        # ========== CAPTURA (CPU) ==========
        ok, frame = cap.read()
        if not ok:
            break

        # ========== PREPROCESO (CPU) ==========
        # Produce un tensor contiguo NCHW float32 (1,3,IM_SIZE,IM_SIZE)
        x_host = preprocess_bgr_cpu(frame, IM_SIZE)

        # Prealoca salida host por frame (puedes moverla fuera del loop y reutilizar)
        y_host = np.empty(st["out_shape"], dtype=st["out_dtype"])

        # ========== H→D (PCIe) ==========
        # Copia sincrónica: el host queda libre de inmediato; el dato ya está en VRAM
        cuda.memcpy_htod(st["d_in"], x_host)

        # ========== INFERENCIA (GPU, async en stream) ==========
        # Enlaza direcciones de device (no mueve datos)
        st["ctx"].set_tensor_address(st["inp_name"], int(st["d_in"]))
        st["ctx"].set_tensor_address(st["out_name"], int(st["d_out"]))

        # Lanza la red optimizada en la stream (no bloquea CPU)
        st["ctx"].execute_async_v3(stream_handle=st["stream"].handle)

        # ========== D→H (PCIe, async) + SYNC ==========
        # Encola la copia de salida y espera a que termine toda la cola de la stream
        cuda.memcpy_dtoh_async(y_host, st["d_out"], st["stream"])
        st["stream"].synchronize()  # a partir de aquí y_host está listo

        # ========== POST-PROCESO (CPU) ==========
        # y_host: [1, C, H, W] (multiclase) o [1, 1, H, W] (binaria)
        logits = y_host[0]  # [C,H,W] o [1,H,W]
        if logits.ndim == 3 and logits.shape[0] > 1:
            pred = logits.argmax(axis=0).astype(np.uint8)  # multiclase
        else:
            pred = (logits.squeeze() > 0.5).astype(np.uint8)  # binaria

        # Colorear y superponer
        overlay = overlay_mask(frame, colorize_mask(pred), alpha=0.5)

        # ========== HUD (CPU) ==========
        n_frames += 1
        t_now = time.perf_counter()
        fps_avg = n_frames / (t_now - t0 + 1e-9)
        cv2.putText(overlay, f"FPS(avg): {fps_avg:4.1f}", (16, 32), 0, 0.7, (40,255,40), 2)

        if categories is not None:
            uniq = np.unique(pred)[:5]
            labels = [categories[c] if c < len(categories) else str(c) for c in uniq]
            cv2.putText(overlay, f"Clases: {', '.join(labels)}", (16, 64), 0, 0.6, (40,255,40), 2)

        # ========== DISPLAY ==========
        cv2.imshow(window_name, overlay)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    # --- Liberación de recursos ---
    cap.release()
    cv2.destroyAllWindows()
# =========================