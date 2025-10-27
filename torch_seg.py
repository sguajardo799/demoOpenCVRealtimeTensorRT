import time
import cv2
import numpy as np

import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

# ----------------------------
# Constantes y utilidades
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_palette(n=256, seed=0):
    rng = np.random.RandomState(seed)
    pal = (rng.rand(n, 3) * 255).astype(np.uint8)
    pal[0] = np.array([0, 0, 0], np.uint8)  # clase 0 → negro (fondo)
    return pal

PALETTE = make_palette(256)

def preprocess_bgr(frame_bgr, size, device):
    """BGR -> RGB, resize, normaliza imagenet, [1,3,H,W]"""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    x = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1,3,H,W)
    xt = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    return xt

def colorize_mask(mask_uint8):
    """mask [H,W] uint8 -> [H,W,3] uint8"""
    return PALETTE[mask_uint8]

def overlay_mask(bgr, mask_color, alpha=0.5):
    mask_resized = cv2.resize(mask_color, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(bgr, 1.0 - alpha, mask_resized, alpha, 0.0)

# ----------------------------
# Inicialización de modelo
# ----------------------------
def init_model(device, im_size=512):
    """
    Carga DeepLabV3 MobileNetV3 con pesos por defecto (COCO con etiquetas tipo VOC),
    pone eval(), channels_last y activa cudnn autotune.
    """
    #weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    #model = deeplabv3_mobilenet_v3_large(weights=weights).eval().to(device)
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).eval().to(device)
    
    
    model = model.to(memory_format=torch.channels_last)

    categories = weights.meta.get("categories", None)
    return model, categories, im_size

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, categories, IM_SIZE = init_model(device, im_size=512)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Error: No se pudo abrir la cámara (VideoCapture(0)).")

    t_last = time.perf_counter()
    fps_ema = 0.0
    alpha = 0.10
    n_frames = 0
    t_start = t_last

    window_name = "Inferencia (DeepLabV3-MNV3, PyTorch+OpenCV)"
    print("Presiona 'q' para salir.")

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame. Exiting...")
                break

            t_now = time.perf_counter()
            dt = t_now - t_last
            t_last = t_now
            fps_inst = (1.0 / dt) if dt > 0 else 0.0
            fps_ema = fps_ema * (1.0 - alpha) + fps_inst * alpha

            # Preprocesa
            x = preprocess_bgr(frame, size=IM_SIZE, device=device)
            x = x.to(memory_format=torch.channels_last)

            # Inferencia 
            if device == "cuda":
                out = model(x)["out"]  # [1, C, H, W]
            else:
                out = model(x)["out"]

            # Post-proceso
            pred = out.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)  # [H,W]
            mask_color = colorize_mask(pred)
            overlay = overlay_mask(frame, mask_color, alpha=0.5)

            # Métricas
            n_frames += 1
            fps_avg = n_frames / (t_now - t_start + 1e-9)

            # Texto en pantalla
            cv2.putText(overlay, f"Device: {device.upper()}", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 2)
            cv2.putText(overlay, f"FPS(inst):  {fps_inst:5.1f}", (16, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 1)
            cv2.putText(overlay, f"FPS(smooth):{fps_ema:5.1f}", (16, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 1)
            cv2.putText(overlay, f"FPS(avg):   {fps_avg:5.1f}", (16, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 1)

            # Info de clases (opcional, si el peso trae categorías)
            if categories is not None:
                # imprime hasta 5 clases detectadas por presencia (muy básico)
                uniq = np.unique(pred)
                labels = [categories[c] if c < len(categories) else str(c) for c in uniq[:5]]
                cv2.putText(overlay, f"Clases~: {', '.join(labels)}", (16, 116),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 40), 1)

            cv2.imshow(window_name, overlay)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
