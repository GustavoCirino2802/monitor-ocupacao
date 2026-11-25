# Sistema de Monitoramento de Ocupação de Salas

Aplicação em Python que detecta, conta e rastreia pessoas em tempo real (webcam, vídeo ou imagem) e avisa quando a lotação de uma sala é excedida.

O projeto cobre as **duas opções** da atividade:

| Modo | Técnica | O que faz |
|------|---------|-----------|
| `yolo` (padrão) | YOLOv8 pré-treinado (COCO) + rastreamento ByteTrack | Detecta pessoas, mostra caixa, classe, confiança e ID |
| `opencv` | Pré-processamento + subtração de fundo MOG2 + contornos | Detecta regiões em movimento sem rede neural |

Em ambos os modos o sistema exibe em tempo real: quantidade atual, % da capacidade, pico, média, área detectada, FPS e status (OK / LOTAÇÃO EXCEDIDA).

## Estrutura

```
monitor_ocupacao/
├── main.py                 # ponto de entrada (laço principal)
├── gerar_video_teste.py    # gera um vídeo sintético para testes
├── requirements.txt
├── src/
│   ├── config.py           # parâmetros padrão
│   ├── fonte_video.py      # captura: webcam / vídeo / imagem
│   ├── preprocessamento.py # redimensionamento, CLAHE, blur, morfologia
│   ├── detectores.py       # DetectorYOLO e DetectorOpenCV
│   ├── estatisticas.py     # métricas, log CSV e resumo
│   └── visualizacao.py     # caixas e painel de métricas
├── amostras/               # vídeo de teste
├── resultados/             # saídas geradas (CSV, resumo, vídeo, imagens)
└── docs/                   # relatório, roteiro do vídeo, explicação do código
```

## Instalação

Requer Python 3.9 ou superior.

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Na primeira execução do modo YOLO o arquivo `yolov8n.pt` (~6 MB) é baixado automaticamente.

## Execução

```bash
# Webcam, modo YOLO, capacidade padrão (5 pessoas)
python main.py

# Vídeo, capacidade de 8 pessoas, gravando o resultado
python main.py --fonte caminho/video.mp4 --capacidade 8 --salvar

# Imagem estática (salva resultados/imagem_processada.jpg)
python main.py --fonte caminho/foto.jpg

# Detectar todas as 80 classes do COCO, não só pessoas
python main.py --todas-classes

# Modo OpenCV com o vídeo de teste e a máscara binária visível
python gerar_video_teste.py
python main.py --modo opencv --fonte amostras/video_teste.mp4 --capacidade 3 --mostrar-mascara
```

### Teclas durante a execução

| Tecla | Ação |
|-------|------|
| `q` | sair |
| `p` | pausar / continuar |
| `s` | salvar o frame atual em `resultados/` |

### Parâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--fonte` | `0` | índice da webcam ou caminho de vídeo/imagem |
| `--modo` | `yolo` | `yolo` ou `opencv` |
| `--modelo` | `yolov8n.pt` | pesos YOLO (ex.: `yolov8s.pt`, mais preciso e mais lento) |
| `--confianca` | `0.40` | confiança mínima das detecções |
| `--todas-classes` | — | detecta todas as classes do COCO |
| `--sem-rastreio` | — | desativa os IDs de rastreamento |
| `--capacidade` | `5` | lotação máxima da sala |
| `--area-minima` | `1500` | modo OpenCV: área mínima do contorno (px²) |
| `--salvar` | — | grava `resultados/saida_processada.mp4` |
| `--sem-janela` | — | roda sem interface gráfica (implica `--salvar`) |
| `--mostrar-mascara` | — | modo OpenCV: mostra a máscara de segmentação |

## Saídas

Ao final de cada execução, em `resultados/`:

- `log_AAAAMMDD_HHMMSS.csv` — uma linha por segundo com quantidade, área, % de ocupação, alerta e FPS (separador `;`, abre direto no Excel);
- `resumo_AAAAMMDD_HHMMSS.txt` — duração, FPS médio, média, pico, % do tempo acima da lotação e pessoas distintas;
- `saida_processada.mp4` — vídeo anotado (com `--salvar`);
- `imagem_processada.jpg` — quando a fonte é uma imagem.

## Problemas comuns

- **Webcam não abre:** tente `--fonte 1` ou feche outros programas que estejam usando a câmera.
- **Erro de janela (`cv2.imshow`) em servidor/WSL:** use `--sem-janela`.
- **FPS baixo no modo YOLO:** use o modelo `yolov8n.pt` (padrão) ou `--sem-rastreio`.
