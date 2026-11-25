"""
preprocessamento.py
-------------------
Etapas de pré-processamento de imagem usadas pelos detectores.
"""

import cv2
from src import config


def redimensionar(frame, largura=config.LARGURA_PROCESSAMENTO):
    """Redimensiona mantendo a proporção. Frames menores aceleram o processamento."""
    altura_orig, largura_orig = frame.shape[:2]
    if largura_orig <= largura:
        return frame
    escala = largura / largura_orig
    return cv2.resize(frame, (largura, int(altura_orig * escala)), interpolation=cv2.INTER_AREA)


def preparar_para_segmentacao(frame):
    """
    Equaliza o contraste (CLAHE) apenas no canal de luminosidade (espaço LAB)
    e aplica blur gaussiano. A cor é mantida de propósito: o subtrator MOG2
    usa a informação de cor para diferenciar objetos escuros de sombras.
    Isso reduz o efeito de variações de iluminação e de ruído do sensor.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    equalizado = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(equalizado, config.KERNEL_BLUR, 0)


def limpar_mascara(mascara):
    """
    Pós-processamento morfológico da máscara binária:
    - limiarização remove as sombras (valor 127 no MOG2);
    - abertura elimina pontos isolados;
    - fechamento + dilatação unem partes do mesmo objeto.
    """
    _, binaria = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_grande, iterations=2)
    return cv2.dilate(binaria, kernel, iterations=1)
