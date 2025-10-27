import time
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# =========================
# Constantes y utilidades
# =========================
IM_SIZE = 512
MODEL_PLAN = "model.plan"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_palette(n=256, seed=0):
    rng = np.random.RandomState(seed)
    pal = (rng.rand(n, 3) * 255).astype(np.uint8)
    pal[0] = np.array([0, 0, 0], np.uint8)  # clase 0 → negro
    return pal

PALETTE = make_palette(256)
categories = None  # reemplaza por lista COCO si quieres rotular

def colorize_mask(mask_uint8):
    return PALETTE[mask_uint8]

def overlay_mask(bgr, mask_color, alpha=0.5):
    mask_resized = cv2.resize(mask_color, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(bgr, 1.0 - alpha, mask_resized, alpha, 0.0)

# =========================
# Preprocesado
# =========================
def preprocess_bgr_cpu(frame_bgr, size=IM_SIZE, out_nchw=None):
    """BGR -> RGB -> resize -> normaliza (ImageNet) -> NCHW float32 en [1,3,H,W]"""
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std  = np.asarray(IMAGENET_STD,  dtype=np.float32)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1))  # (3,H,W)

    if out_nchw is None:
        x = np.empty((1, 3, size, size), dtype=np.float32)
    else:
        x = out_nchw  # buffer pagelocked
    x[0, ...] = chw
    return x

def has_cv_cuda():
    """Devuelve True solo si OpenCV fue compilado con CUDA y hay device disponible."""
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

def preprocess_bgr_gpu(frame_bgr, gpu_ctx, out_nchw):
    """Versión con OpenCV CUDA (si está disponible). Descarga y normaliza en host."""
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std  = np.asarray(IMAGENET_STD,  dtype=np.float32)

    gmat = cv2.cuda_GpuMat()
    gmat.upload(frame_bgr, stream=gpu_ctx["cv_stream"])
    gmat = cv2.cuda.cvtColor(gmat, cv2.COLOR_BGR2RGB)
    gmat = cv2.cuda.resize(gmat, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_LINEAR)
    gmat = gmat.convertTo(cv2.CV_32F, stream=gpu_ctx["cv_stream"], scale=1.0/255.0)

    tmp = gmat.download(stream=gpu_ctx["cv_stream"])
    gpu_ctx["cv_stream"].waitForCompletion()

    tmp = (tmp - mean) / std
    chw = np.transpose(tmp, (2, 0, 1))
    out_nchw[0, ...] = chw
    return out_nchw

# =========================
# TensorRT init + buffers
# =========================
def _dims_to_tuple(dims: "trt.Dims"):
    return tuple(int(dims[i]) for i in range(len(dims)))

def init_trt(plan_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(plan_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    # Nombres de tensores (API de tensores)
    inp_name = [n for n in engine if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    out_name = [n for n in engine if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]

    # Shape fija de entrada (1,3,512,512)
    in_shape = (1, 3, IM_SIZE, IM_SIZE)
    if -1 in _dims_to_tuple(engine.get_tensor_shape(inp_name)):
        ctx.set_input_shape(inp_name, in_shape)

    # Shape/dtype de salida (convierte Dims -> tupla)
    out_shape = _dims_to_tuple(ctx.get_tensor_shape(out_name))  # esperado: (1, C, 512, 512)
    out_dtype = trt.nptype(engine.get_tensor_dtype(out_name))
    C = out_shape[1]

    # Buffers host paginados (pinned) - doble buffer
    h_in  = [cuda.pagelocked_empty(in_shape,  dtype=np.float32),
             cuda.pagelocked_empty(in_shape,  dtype=np.float32)]
    h_out = [cuda.pagelocked_empty(out_shape, dtype=out_dtype),
             cuda.pagelocked_empty(out_shape, dtype=out_dtype)]

    # Buffers device - doble buffer
    d_in  = [cuda.mem_alloc(int(np.prod(in_shape)  * np.dtype(np.float32).itemsize)) for _ in range(2)]
    d_out = [cuda.mem_alloc(int(np.prod(out_shape) * np.dtype(out_dtype).itemsize))  for _ in range(2)]

    # Stream CUDA
    stream = cuda.Stream()

    return {
        "engine": engine, "ctx": ctx,
        "inp_name": inp_name, "out_name": out_name,
        "in_shape": in_shape, "out_shape": out_shape, "out_dtype": out_dtype, "C": C,
        "h_in": h_in, "h_out": h_out, "d_in": d_in, "d_out": d_out,
        "stream": stream
    }

# =========================
# Main (video)
# =========================
if __name__ == "__main__":
    cv2.setNumThreads(1)  # menos overhead en CPU
    st = init_trt(MODEL_PLAN)

    # Detección robusta: si OpenCV no tiene CUDA, forzamos CPU
    use_cv_cuda = has_cv_cuda()
    gpu_ctx = None
    if use_cv_cuda:
        try:
            gpu_ctx = {"cv_stream": cv2.cuda.Stream()}
        except Exception:
            use_cv_cuda = False
            gpu_ctx = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    print("Presiona 'q' para salir.")
    t_last = time.perf_counter()
    t0 = t_last
    fps_ema, alpha = 0.0, 0.10
    n_frames = 0
    window_name = "TRT Seg (pipeline)"

    # Prefill primer frame en buffer 0
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("No se pudo leer el primer frame.")
    if use_cv_cuda:
        preprocess_bgr_gpu(frame, gpu_ctx, st["h_in"][0])
    else:
        preprocess_bgr_cpu(frame, IM_SIZE, st["h_in"][0])

    buf = 0  # doble buffer
    while True:
        nxt = 1 - buf

        # Captura + preprocesa siguiente frame en host pinned (nxt)
        ok, frame = cap.read()
        if not ok:
            break
        if use_cv_cuda:
            preprocess_bgr_gpu(frame, gpu_ctx, st["h_in"][nxt])
        else:
            preprocess_bgr_cpu(frame, IM_SIZE, st["h_in"][nxt])

        # Subir input actual
        cuda.memcpy_htod_async(st["d_in"][buf], st["h_in"][buf], st["stream"])

        # Direcciones por nombre (API v3) con el buffer actual
        st["ctx"].set_tensor_address(st["inp_name"], int(st["d_in"][buf]))
        st["ctx"].set_tensor_address(st["out_name"], int(st["d_out"][buf]))

        # Ejecutar
        st["ctx"].execute_async_v3(stream_handle=st["stream"].handle)

        # Bajar salida
        cuda.memcpy_dtoh_async(st["h_out"][buf], st["d_out"][buf], st["stream"])
        st["stream"].synchronize()

        # Post-proceso
        out_host = st["h_out"][buf]  # [1,C,512,512] o [1,1,512,512]
        if st["C"] > 1:
            pred = out_host[0].argmax(axis=0).astype(np.uint8)   # [H,W]
        else:
            pred = (out_host[0, 0] > 0.5).astype(np.uint8)

        overlay = overlay_mask(frame, colorize_mask(pred), alpha=0.5)

        # FPS
        n_frames += 1
        t_now = time.perf_counter()
        dt = t_now - t_last
        t_last = t_now
        fps_inst = (1.0 / dt) if dt > 0 else 0.0
        fps_ema = fps_ema * (1.0 - alpha) + fps_inst * alpha
        fps_avg = n_frames / (t_now - t0 + 1e-9)

        # HUD
        cv2.putText(overlay, f"FPS(inst):  {fps_inst:4.1f}",  (16, 32), 0, 0.6, (40,255,40), 2)
        cv2.putText(overlay, f"FPS(smooth):{fps_ema:4.1f}",  (16, 60), 0, 0.5, (40,255,40), 1)
        cv2.putText(overlay, f"FPS(avg):   {fps_avg:4.1f}",  (16, 80), 0, 0.5, (40,255,40), 1)

        if categories is not None:
            uniq = np.unique(pred)[:5]
            labels = [categories[c] if c < len(categories) else str(c) for c in uniq]
            cv2.putText(overlay, f"Clases~: {', '.join(labels)}", (16, 100), 0, 0.5, (40,255,40), 1)

        cv2.imshow(window_name, overlay)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        # alternar buffer
        buf = nxt

    cap.release()
    cv2.destroyAllWindows()
