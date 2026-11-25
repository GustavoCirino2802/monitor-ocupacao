"""
detectores.py
-------------
Dois detectores com a mesma interface `detectar(frame) -> list[Deteccao]`:

1. DetectorYOLO    -> Opção 2 da atividade (modelo YOLO pré-treinado).
2. DetectorOpenCV  -> Opção 1 da atividade (subtração de fundo + contornos).

Assim o restante do sistema (métricas, desenho, log) não precisa saber qual
técnica está sendo usada.
"""

from dataclasses import dataclass
from typing import Optional

import cv2

from src import config
from src.preprocessamento import preparar_para_segmentacao, limpar_mascara


@dataclass
class Deteccao:
    """Um objeto detectado no frame."""
    x1: int
    y1: int
    x2: int
    y2: int
    classe: str
    confianca: float
    id_rastreio: Optional[int] = None   # preenchido quando o rastreamento está ativo

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def centro(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class DetectorYOLO:
    """Detecção (e rastreamento opcional) de pessoas com Ultralytics YOLO."""

    def __init__(self, modelo=config.MODELO_YOLO, confianca=config.CONFIANCA_MINIMA,
                 classes=(config.CLASSE_PESSOA,), rastrear=True):
        # Import tardio: o modo OpenCV funciona mesmo sem a ultralytics instalada
        from ultralytics import YOLO

        self.modelo = YOLO(modelo)          # baixa os pesos na primeira execução
        self.confianca = confianca
        self.classes = list(classes) if classes else None  # None = todas as 80 classes
        self.rastrear = rastrear
        self.nomes = self.modelo.names      # dicionário {id: "nome da classe"}

    def detectar(self, frame):
        parametros = dict(conf=self.confianca, classes=self.classes, verbose=False)
        if self.rastrear:
            # ByteTrack mantém o mesmo ID para a mesma pessoa entre frames
            resultados = self.modelo.track(frame, persist=True, tracker="bytetrack.yaml", **parametros)
        else:
            resultados = self.modelo(frame, **parametros)

        caixas = resultados[0].boxes
        if caixas is None or len(caixas) == 0:
            return []

        coords = caixas.xyxy.cpu().numpy().astype(int)
        confs = caixas.conf.cpu().numpy()
        clss = caixas.cls.cpu().numpy().astype(int)
        ids = caixas.id.cpu().numpy().astype(int) if caixas.id is not None else [None] * len(coords)

        return [
            Deteccao(int(x1), int(y1), int(x2), int(y2), self.nomes[c], float(cf),
                     int(i) if i is not None else None)
            for (x1, y1, x2, y2), cf, c, i in zip(coords, confs, clss, ids)
        ]


class DetectorOpenCV:
    """
    Detecção clássica de objetos em movimento:
    pré-processamento -> subtração de fundo (MOG2) -> morfologia -> contornos.
    Não identifica a classe do objeto, apenas regiões em movimento.
    """

    def __init__(self, area_minima=config.AREA_MINIMA_CONTORNO):
        self.area_minima = area_minima
        self.subtrator = cv2.createBackgroundSubtractorMOG2(
            history=config.HISTORICO_MOG2,
            varThreshold=config.LIMIAR_MOG2,
            detectShadows=True,
        )
        self.ultima_mascara = None  # guardada para exibição opcional

    def detectar(self, frame):
        preparado = preparar_para_segmentacao(frame)
        mascara = self.subtrator.apply(preparado)
        mascara = limpar_mascara(mascara)
        self.ultima_mascara = mascara

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        deteccoes = []
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area < self.area_minima:
                continue  # descarta ruído
            x, y, w, h = cv2.boundingRect(contorno)
            # "Confiança" aproximada: quanto do retângulo é preenchido pelo contorno
            preenchimento = area / float(w * h)
            deteccoes.append(Deteccao(x, y, x + w, y + h, "objeto", round(preenchimento, 2)))
        return deteccoes
