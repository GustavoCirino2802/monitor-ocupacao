"""
config.py
---------
Parâmetros padrão do sistema. Todos podem ser sobrescritos pela linha de
comando (ver main.py), mas ficam centralizados aqui para facilitar ajustes.
"""

# ---------------- YOLO ----------------
MODELO_YOLO = "yolov8n.pt"       # modelo pré-treinado (COCO, 80 classes); baixado automaticamente
CONFIANCA_MINIMA = 0.40          # descarta detecções com confiança abaixo deste valor
CLASSE_PESSOA = 0                # no dataset COCO, a classe 0 é "person"

# ---------------- OpenCV (modo clássico) ----------------
LARGURA_PROCESSAMENTO = 640      # frames são redimensionados para esta largura
KERNEL_BLUR = (5, 5)             # tamanho do filtro gaussiano (redução de ruído)
AREA_MINIMA_CONTORNO = 1500      # contornos menores (px²) são considerados ruído
HISTORICO_MOG2 = 300             # nº de frames usados para modelar o fundo
LIMIAR_MOG2 = 40                 # sensibilidade do subtrator de fundo

# ---------------- Regras de ocupação ----------------
CAPACIDADE_SALA = 5              # lotação máxima; acima disso o sistema emite alerta
INTERVALO_LOG_SEG = 1.0          # de quanto em quanto tempo gravar uma linha no CSV

# ---------------- Saída ----------------
PASTA_RESULTADOS = "resultados"

# Cores em BGR (padrão do OpenCV)
COR_OK = (0, 200, 0)
COR_ALERTA = (0, 0, 255)
COR_CAIXA = (255, 160, 0)
COR_TEXTO = (255, 255, 255)
COR_PAINEL = (30, 30, 30)
