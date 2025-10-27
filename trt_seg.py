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

def make_palette(n=256, seed=0):
    rng = np.random.RandomState(seed)
    pal = (rng.rand(n, 3) * 255).astype(np.uint8)
    pal[0] = np.array([0, 0, 0], np.uint8)
    return pal

PALETTE = make_palette(256)
categories = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

def colorize_mask(mask_uint8):
    return PALETTE[mask_uint8]

def overlay_mask(bgr, mask_color, alpha=0.5):
    mask_resized = cv2.resize(mask_color, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(bgr, 1.0 - alpha, mask_resized, alpha, 0.0)

# =========================
# Preprocesado CPU simple
# =========================
def preprocess_bgr_cpu_into(frame_bgr, out_array, size=IM_SIZE):
    # out_array: (1,3,H,W) con dtype = st["inp_dtype"] (np.float16 si el plan es FP16)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std  = np.asarray(IMAGENET_STD,  dtype=np.float32)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std                      # float32 aquí por estabilidad
    chw = np.transpose(rgb, (2, 0, 1))            # (3,H,W) (vista)
    # copiar a buffer contiguo y convertir a dtype de input (fp16)
    out_array[0] = chw.astype(out_array.dtype, copy=False)

# =========================
# TensorRT init (sin doble buffer)
# =========================
def load_engine_and_alloc(plan_path):
    """
    Carga un engine TensorRT desde un .plan, crea el contexto de ejecución y
    reserva UNA vez la memoria de device para entrada y salida. Devuelve
    los identificadores clave para ejecutar inferencia cuadro a cuadro
    con copias simples H->D, ejecución async y D->H.

    Suposiciones:
        - La ENTRADA del modelo es fija y coincide con el preprocesado:
          (1, 3, IM_SIZE, IM_SIZE) en float32.
        - La SALIDA del modelo está definida en el plan; su shape y dtype se
          leen del engine/contexto (p.ej., segmentación 1xCxHxW o 1x1xHxW).

    Parámetros
    ----------
    plan_path : str
        Ruta al archivo TensorRT serializado (*.plan).

    Retorna
    -------
    dict
        Diccionario con los objetos necesarios para ejecutar inferencia:
        - engine : trt.ICudaEngine
            Engine deserializado. Contiene la red optimizada y la
            descripción de I/O.
        - ctx : trt.IExecutionContext
            Contexto de ejecución asociado al engine. No es thread-safe.
        - inp_name : str
            Nombre del tensor de entrada (API de tensores).
        - out_name : str
            Nombre del tensor de salida principal.
        - out_shape : tuple[int, ...]
            Forma de la salida tal como la reporta el engine/contexto.
            Se usa para dimensionar buffers host y post-procesar.
        - out_dtype : numpy.dtype
            Tipo de dato real de la salida (p.ej., np.float32, np.float16, np.int8).
        - d_in : pycuda.driver.DeviceAllocation
            Único buffer en device para la entrada. Tamaño: 1*3*IM_SIZE*IM_SIZE*4 bytes.
        - d_out : pycuda.driver.DeviceAllocation
            Único buffer en device para la salida. Tamaño = prod(out_shape) * sizeof(out_dtype).
        - stream : pycuda.driver.Stream
            Stream CUDA para copias async y ejecución (`execute_async_v3`).
    """
    logger = trt.Logger(trt.Logger.WARNING)
    with open(plan_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    inp_name = next(n for n in engine if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    out_name = next(n for n in engine if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

    # Dtypes reales del engine (si el plan está en FP16, esto será float16)
    inp_dtype = trt.nptype(engine.get_tensor_dtype(inp_name))
    out_dtype = trt.nptype(engine.get_tensor_dtype(out_name))

    # Entrada fija por diseño del preprocess
    in_shape = (1, 3, IM_SIZE, IM_SIZE)
    bytes_in = int(np.prod(in_shape)) * np.dtype(inp_dtype).itemsize
    d_in = cuda.mem_alloc(bytes_in)

    # Salida desde el engine/contexto
    out_dims = engine.get_tensor_shape(out_name)
    out_shape = tuple(int(out_dims[i]) for i in range(len(out_dims)))
    bytes_out = int(np.prod(out_shape)) * np.dtype(out_dtype).itemsize
    d_out = cuda.mem_alloc(bytes_out)

    stream = cuda.Stream()
    return {
        "engine": engine, "ctx": ctx,
        "inp_name": inp_name, "out_name": out_name,
        "in_shape": in_shape, "out_shape": out_shape,
        "inp_dtype": inp_dtype, "out_dtype": out_dtype,
        "d_in": d_in, "d_out": d_out, "stream": stream
    }


# =========================
# Main (video)
# =========================
if __name__ == "__main__":
    cv2.setNumThreads(0)  # deja que OpenCV decida; prueba 0 u 8 y mide

    st = load_engine_and_alloc(MODEL_PLAN)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    # Reusar host arrays (1 vez)
    x_host = np.empty(st["in_shape"], dtype=st["inp_dtype"])     # input host (fp16)
    y_host = np.empty(st["out_shape"], dtype=st["out_dtype"])    # output host (fp16 si plan)
    # “Pinnea” estas mismas memorias (no creas nuevas)
    cuda.register_host_memory(x_host)
    cuda.register_host_memory(y_host)

    t0, n_frames = time.perf_counter(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        preprocess_bgr_cpu_into(frame, x_host, IM_SIZE)          # escribe dentro de x_host (contiguo)

        # H->D (async)
        cuda.memcpy_htod_async(st["d_in"], x_host, st["stream"])

        # Ejecuta
        st["ctx"].set_tensor_address(st["inp_name"], int(st["d_in"]))
        st["ctx"].set_tensor_address(st["out_name"], int(st["d_out"]))
        st["ctx"].execute_async_v3(stream_handle=st["stream"].handle)

        # D->H (async) y sincroniza sólo antes de usar y_host
        cuda.memcpy_dtoh_async(y_host, st["d_out"], st["stream"])
        st["stream"].synchronize()

        # Post (argmax funciona sobre float16 sin problema)
        logits = y_host[0]
        pred = logits.argmax(axis=0).astype(np.uint8) if logits.shape[0] > 1 else (logits.squeeze() > 0.5).astype(np.uint8)

        overlay = overlay_mask(frame, colorize_mask(pred), alpha=0.5)
        n_frames += 1
        fps = n_frames / (time.perf_counter() - t0 + 1e-9)
        cv2.putText(overlay, f"FPS: {fps:4.1f}", (16, 32), 0, 0.7, (40,255,40), 2)

        cv2.imshow("TRT Seg (fp16+pinned)", overlay)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
