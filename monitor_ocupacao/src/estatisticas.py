"""
estatisticas.py
---------------
Métricas da execução: contagem por frame, pico, média, ocupação acima da
capacidade, área detectada, pessoas únicas (via ID de rastreio) e FPS.
Também grava um log em CSV e um resumo final em TXT.
"""

import csv
import os
import time
from collections import Counter
from datetime import datetime

from src import config


class Estatisticas:
    def __init__(self, capacidade=config.CAPACIDADE_SALA, pasta=config.PASTA_RESULTADOS):
        self.capacidade = capacidade
        self.pasta = pasta
        os.makedirs(pasta, exist_ok=True)

        carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.caminho_csv = os.path.join(pasta, f"log_{carimbo}.csv")
        self.caminho_resumo = os.path.join(pasta, f"resumo_{carimbo}.txt")
        self._arquivo_csv = open(self.caminho_csv, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._arquivo_csv, delimiter=";")
        self._csv.writerow(["data_hora", "frame", "quantidade", "area_total_px", "ocupacao_%", "alerta", "fps"])

        self.inicio = time.time()
        self._ultimo_log = 0.0
        self._ultimo_frame_t = time.time()
        self.fps = 0.0

        self.frames = 0
        self.soma_contagens = 0
        self.pico = 0
        self.frames_em_alerta = 0
        self.ids_unicos = set()
        self.contagem_classes = Counter()
        self.atual = 0
        self.area_atual = 0

    def atualizar(self, deteccoes):
        """Atualiza todas as métricas com as detecções de um novo frame."""
        agora = time.time()
        dt = agora - self._ultimo_frame_t
        self._ultimo_frame_t = agora
        if dt > 0:  # média móvel exponencial para o FPS não "pular" tanto
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt

        self.frames += 1
        self.atual = len(deteccoes)
        self.area_atual = sum(d.area for d in deteccoes)
        self.soma_contagens += self.atual
        self.pico = max(self.pico, self.atual)
        if self.em_alerta:
            self.frames_em_alerta += 1
        for d in deteccoes:
            self.contagem_classes[d.classe] += 1
            if d.id_rastreio is not None:
                self.ids_unicos.add(d.id_rastreio)

        # Grava no CSV em intervalos fixos (evita arquivos gigantes)
        if agora - self._ultimo_log >= config.INTERVALO_LOG_SEG:
            self._ultimo_log = agora
            self._csv.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.frames, self.atual,
                self.area_atual, f"{self.ocupacao_percentual:.0f}", int(self.em_alerta), f"{self.fps:.1f}",
            ])

    # ---- propriedades derivadas ----
    @property
    def ocupacao_percentual(self):
        return 100.0 * self.atual / self.capacidade if self.capacidade else 0.0

    @property
    def em_alerta(self):
        return self.atual > self.capacidade

    @property
    def media(self):
        return self.soma_contagens / self.frames if self.frames else 0.0

    def finalizar(self):
        """Fecha o CSV, grava o resumo e devolve o texto do resumo."""
        self._arquivo_csv.close()
        duracao = time.time() - self.inicio
        perc_alerta = 100.0 * self.frames_em_alerta / self.frames if self.frames else 0.0
        linhas = [
            "===== RESUMO DA EXECUÇÃO =====",
            f"Duração ................: {duracao:.1f} s",
            f"Frames processados .....: {self.frames}",
            f"FPS médio ..............: {self.frames / duracao if duracao else 0:.1f}",
            f"Capacidade da sala .....: {self.capacidade}",
            f"Média de objetos/frame .: {self.media:.2f}",
            f"Pico simultâneo ........: {self.pico}",
            f"Tempo acima da lotação .: {perc_alerta:.1f}% dos frames",
        ]
        if self.ids_unicos:
            linhas.append(f"Pessoas distintas (IDs) : {len(self.ids_unicos)}")
        if self.contagem_classes:
            linhas.append("Detecções por classe ...: " +
                          ", ".join(f"{c}={n}" for c, n in self.contagem_classes.most_common()))
        linhas.append(f"Log detalhado ..........: {self.caminho_csv}")
        texto = "\n".join(linhas)
        with open(self.caminho_resumo, "w", encoding="utf-8") as f:
            f.write(texto + "\n")
        return texto
