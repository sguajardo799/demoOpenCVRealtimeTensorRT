import time
import cv2
import numpy as np

import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

# =========================
# Constantes y utilidades
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def make_palette(n: int = 256, seed: int = 0) -> np.ndarray:
    """
    Genera una paleta indexada (n x 3, uint8) para colorear máscaras.
    Índice 0 se fuerza a negro (fondo).

    Args:
        n: cantidad de colores/clases.
        seed: semilla para reproducibilidad.

    Returns:
        np.ndarray de shape (n, 3) con valores [0..255].
    """
    rng = np.random.RandomState(seed)
    pal = (rng.rand(n, 3) * 255).astype(np.uint8)
    pal[0] = np.array([0, 0, 0], np.uint8)
    return pal


PALETTE = make_palette(256)


def preprocess_bgr(frame_bgr: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    """
    Preprocesa un frame BGR para DeepLabV3:
    BGR→RGB, resize, normaliza (ImageNet), a NCHW y Tensor float32 en `device`.

    Notas de memoria:
      - NumPy genera (1,3,H,W) contiguo; luego `torch.from_numpy` comparte
        memoria y `.to(device)` copia a VRAM si `device='cuda'`.
      - Convertimos a `channels_last` antes de la inferencia para que cuDNN
        escoja kernels NHWC-friendly.

    Args:
        frame_bgr: imagen BGR uint8 (H,W,3).
        size: lado (cuadrado) de entrada del modelo.
        device: 'cuda' o 'cpu'.

    Returns:
        Tensor torch float32 (1,3,size,size) en `device`, channels_last.
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    x = np.transpose(rgb, (2, 0, 1))[None, ...]   # (1,3,H,W) contiguo

    xt = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    # Ajuste de formato de memoria para kernels optimizados
    xt = xt.to(memory_format=torch.channels_last)
    return xt


def colorize_mask(mask_uint8: np.ndarray) -> np.ndarray:
    """
    Colorea una máscara de clases (H,W) usando PALETTE.

    Args:
        mask_uint8: máscara de clases uint8 (H,W).

    Returns:
        Imagen RGB (H,W,3) uint8.
    """
    return PALETTE[mask_uint8]


def overlay_mask(bgr: np.ndarray, mask_color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Superpone la máscara coloreada sobre el frame BGR original.

    Args:
        bgr: imagen base BGR (H,W,3).
        mask_color: imagen RGB coloreada (h,w,3).
        alpha: opacidad de la máscara.

    Returns:
        BGR (H,W,3) con overlay.
    """
    mask_resized = cv2.resize(mask_color, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(bgr, 1.0 - alpha, mask_resized, alpha, 0.0)


# =========================
# Inicialización de modelo
# =========================
def init_model(device: torch.device, im_size: int = 512):
    """
    Carga DeepLabV3-ResNet50 preentrenado (weights DEFAULT), lo pasa a eval(),
    ajusta formato de memoria y activa cuDNN autotune para entrada fija.

    Args:
        device: 'cuda' o 'cpu'.
        im_size: tamaño de entrada (cuadrado).

    Returns:
        model: nn.Module listo para inferencia en `device`.
        categories: lista de clases según los weights (VOC); puede ser None.
        im_size: int, se retorna para mantener contrato con el caller.
    """
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).eval().to(device)

    # Preferencia NHWC para kernels; no cambia semántica del tensor
    model = model.to(memory_format=torch.channels_last)

    # Si la resolución y el modelo son fijos, cuDNN puede auto-optimizar
    torch.backends.cudnn.benchmark = True

    categories = weights.meta.get("categories", None)
    return model, categories, im_size


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Dispositivo (usa CUDA si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, categories, IM_SIZE = init_model(device, im_size=512)

    # Captura de cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Error: No se pudo abrir la cámara (VideoCapture(0)).")

    t_last = time.perf_counter()
    fps_ema = 0.0
    alpha = 0.10
    n_frames = 0
    t_start = t_last

    window_name = "Inferencia (DeepLabV3-ResNet50, PyTorch+OpenCV)"
    print("Presiona 'q' para salir.")

    # Modo inferencia: desactiva grad y optimiza algunos paths internos
    with torch.inference_mode():
        # (Opcional) AMP para FP16 en GPU. Comentar si el GPU no gana con fp16.
        autocast_enabled = device.type == "cuda"
        scaler = None  # no se usa en inferencia pura

        while True:
            # ========== CAPTURA (CPU) ==========
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame. Exiting...")
                break

            # Métrica de tiempo instantáneo
            t_now = time.perf_counter()
            dt = t_now - t_last
            t_last = t_now
            fps_inst = (1.0 / dt) if dt > 0 else 0.0
            fps_ema = fps_ema * (1.0 - alpha) + fps_inst * alpha

            # ========== PREPROCESO (CPU→GPU con .to(device)) ==========
            x = preprocess_bgr(frame, size=IM_SIZE, device=device)

            # ========== INFERENCIA (GPU/CPU) ==========
            if autocast_enabled:
                # fp16 en GPU: reduce BW y puede acelerar el backbone
                with torch.cuda.amp.autocast():
                    out = model(x)["out"]  # [1, C, H, W]
            else:
                out = model(x)["out"]

            # ========== POST-PROCESO (GPU→CPU) ==========
            # Argmax en GPU y copia mínima a CPU (H,W) u8
            pred = out.argmax(dim=1).squeeze(0).to("cpu", dtype=torch.uint8).numpy()
            mask_color = colorize_mask(pred)
            overlay = overlay_mask(frame, mask_color, alpha=0.5)

            # Métricas de FPS
            n_frames += 1
            fps_avg = n_frames / (t_now - t_start + 1e-9)

            # HUD
            cv2.putText(overlay, f"Device: {device.type.upper()}", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 2)
            cv2.putText(overlay, f"FPS(inst):  {fps_inst:5.1f}", (16, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 1)
            cv2.putText(overlay, f"FPS(smooth):{fps_ema:5.1f}", (16, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 1)
            cv2.putText(overlay, f"FPS(avg):   {fps_avg:5.1f}", (16, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 1)

            # Clases (si los weights traen categorías; VOC en torchvision)
            if categories is not None:
                uniq = np.unique(pred)
                labels = [categories[c] if c < len(categories) else str(c) for c in uniq[:5]]
                cv2.putText(overlay, f"Clases~: {', '.join(labels)}", (16, 116),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 1)

            # Display
            cv2.imshow(window_name, overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
