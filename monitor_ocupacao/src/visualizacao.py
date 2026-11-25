"""
visualizacao.py
---------------
Desenho dos resultados sobre o frame: bounding boxes com classe e confiança
e um painel (HUD) com as métricas em tempo real.
"""

import cv2
from src import config

FONTE = cv2.FONT_HERSHEY_SIMPLEX


def desenhar_deteccoes(frame, deteccoes):
    """Desenha a caixa, o rótulo (classe + confiança + ID) e o centro de cada objeto."""
    for d in deteccoes:
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), config.COR_CAIXA, 2)

        rotulo = f"{d.classe} {d.confianca:.0%}"
        if d.id_rastreio is not None:
            rotulo = f"#{d.id_rastreio} " + rotulo

        (tw, th), _ = cv2.getTextSize(rotulo, FONTE, 0.5, 1)
        topo = max(d.y1, th + 6)
        cv2.rectangle(frame, (d.x1, topo - th - 6), (d.x1 + tw + 6, topo), config.COR_CAIXA, -1)
        cv2.putText(frame, rotulo, (d.x1 + 3, topo - 4), FONTE, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, d.centro, 3, config.COR_CAIXA, -1)
    return frame


def desenhar_painel(frame, est, modo):
    """Painel semitransparente no canto superior esquerdo com as métricas."""
    cor_status = config.COR_ALERTA if est.em_alerta else config.COR_OK
    status = "LOTACAO EXCEDIDA" if est.em_alerta else "OK"
    linhas = [
        f"Modo: {modo.upper()}",
        f"Detectados agora: {est.atual}",
        f"Capacidade: {est.capacidade}  ({est.ocupacao_percentual:.0f}%)",
        f"Pico: {est.pico}   Media: {est.media:.1f}",
        f"Area detectada: {est.area_atual} px2",
        f"FPS: {est.fps:.1f}",
    ]
    if est.ids_unicos:
        linhas.append(f"Pessoas distintas: {len(est.ids_unicos)}")

    altura = 22 * len(linhas) + 44
    sobreposicao = frame.copy()
    cv2.rectangle(sobreposicao, (8, 8), (300, 8 + altura), config.COR_PAINEL, -1)
    cv2.addWeighted(sobreposicao, 0.65, frame, 0.35, 0, frame)

    for i, texto in enumerate(linhas):
        cv2.putText(frame, texto, (16, 30 + 22 * i), FONTE, 0.5, config.COR_TEXTO, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Status: {status}", (16, 30 + 22 * len(linhas) + 6),
                FONTE, 0.6, cor_status, 2, cv2.LINE_AA)

    # Borda vermelha no frame inteiro quando a lotação é excedida
    if est.em_alerta:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), config.COR_ALERTA, 6)
    return frame
