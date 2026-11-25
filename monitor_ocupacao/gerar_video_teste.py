"""
gerar_video_teste.py
--------------------
Gera um vídeo sintético (amostras/video_teste.mp4) com formas se movendo
sobre um fundo fixo. Serve para testar o modo OpenCV sem precisar de webcam.

Uso:
    python gerar_video_teste.py
"""

import os
import cv2
import numpy as np

LARGURA, ALTURA, FPS, SEGUNDOS = 640, 360, 25, 12


def main():
    os.makedirs("amostras", exist_ok=True)
    caminho = os.path.join("amostras", "video_teste.mp4")
    gravador = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (LARGURA, ALTURA))

    # Fundo "sala": piso + parede com uma textura leve
    rng = np.random.default_rng(42)
    fundo = np.full((ALTURA, LARGURA, 3), (180, 175, 170), np.uint8)
    fundo[ALTURA // 2:] = (120, 110, 100)
    ruido_fixo = rng.integers(-8, 8, fundo.shape, dtype=np.int16)
    fundo = np.clip(fundo.astype(np.int16) + ruido_fixo, 0, 255).astype(np.uint8)

    # Cada objeto: (x inicial, y, velocidade, largura, altura, cor, frame em que entra)
    objetos = [
        (-60, 150, 4, 45, 110, (60, 60, 200), 0),
        (LARGURA, 190, -3, 50, 120, (200, 80, 40), 30),
        (-60, 120, 5, 40, 100, (40, 160, 40), 70),
        (LARGURA, 170, -4, 55, 125, (30, 30, 30), 110),
        (-60, 200, 3, 45, 115, (150, 40, 150), 150),
        (LARGURA, 140, -5, 40, 105, (0, 140, 220), 180),
        (-60, 160, 4, 50, 110, (90, 90, 90), 200),
    ]

    for n in range(FPS * SEGUNDOS):
        frame = fundo.copy()
        for x0, y, v, w, h, cor, entrada in objetos:
            if n < entrada:
                continue
            x = x0 + v * (n - entrada)
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, -1)            # "corpo"
            cv2.circle(frame, (x + w // 2, y - 18), 18, cor, -1)             # "cabeça"
        # pequeno ruído de sensor a cada frame
        ruido = rng.integers(-3, 3, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
        gravador.write(frame)

    gravador.release()
    print(f"Vídeo de teste gerado em {caminho}")


if __name__ == "__main__":
    main()
