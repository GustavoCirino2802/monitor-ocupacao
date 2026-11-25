"""
fonte_video.py
--------------
Captura de imagem/vídeo. Abstrai as três fontes possíveis (webcam, arquivo de
vídeo e imagem estática) atrás de uma mesma interface: um iterador de frames.
"""

import os
import cv2

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FonteVideo:
    """Fornece frames de uma webcam, de um vídeo ou de uma imagem."""

    def __init__(self, origem):
        # Se a origem for um número ("0", "1"...), tratamos como índice de webcam
        self.origem = int(origem) if str(origem).isdigit() else origem
        self.eh_imagem = (
            isinstance(self.origem, str)
            and os.path.splitext(self.origem)[1].lower() in EXTENSOES_IMAGEM
        )
        self.cap = None
        self.imagem = None

        if self.eh_imagem:
            self.imagem = cv2.imread(self.origem)
            if self.imagem is None:
                raise FileNotFoundError(f"Não foi possível abrir a imagem: {self.origem}")
        else:
            self.cap = cv2.VideoCapture(self.origem)
            if not self.cap.isOpened():
                raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {self.origem}")

    @property
    def fps(self):
        """FPS nominal da fonte (usado ao gravar o vídeo de saída)."""
        if self.cap is None:
            return 1.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def frames(self):
        """Gera os frames um a um. Para imagem, devolve um único frame."""
        if self.eh_imagem:
            yield self.imagem.copy()
            return
        while True:
            ok, frame = self.cap.read()
            if not ok:  # fim do vídeo ou falha na webcam
                break
            yield frame

    def liberar(self):
        """Libera o dispositivo/arquivo de vídeo."""
        if self.cap is not None:
            self.cap.release()
