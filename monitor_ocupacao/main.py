"""
main.py
-------
Ponto de entrada do Sistema de Monitoramento de Ocupação de Salas.

Fluxo principal (para cada frame):
    captura -> pré-processamento -> detecção -> métricas -> desenho -> exibição/gravação

Exemplos:
    python main.py                                  # webcam, modo YOLO
    python main.py --fonte video.mp4 --capacidade 8
    python main.py --fonte foto.jpg
    python main.py --modo opencv --fonte esteira.mp4 --mostrar-mascara
"""

import argparse
import os
import sys

import cv2

from src import config
from src.fonte_video import FonteVideo
from src.preprocessamento import redimensionar
from src.estatisticas import Estatisticas
from src.visualizacao import desenhar_deteccoes, desenhar_painel

JANELA = "Monitor de Ocupacao  [q = sair | p = pausar | s = salvar frame]"


def ler_argumentos():
    """Define e lê os parâmetros de linha de comando."""
    p = argparse.ArgumentParser(description="Detecção e contagem de pessoas/objetos em tempo real.")
    p.add_argument("--fonte", default="0",
                   help="0 = webcam padrão; ou caminho de um vídeo/imagem (padrão: 0)")
    p.add_argument("--modo", choices=["yolo", "opencv"], default="yolo",
                   help="yolo = rede neural pré-treinada; opencv = subtração de fundo (padrão: yolo)")
    p.add_argument("--modelo", default=config.MODELO_YOLO, help="pesos YOLO (padrão: yolov8n.pt)")
    p.add_argument("--confianca", type=float, default=config.CONFIANCA_MINIMA,
                   help="confiança mínima das detecções YOLO (padrão: 0.40)")
    p.add_argument("--todas-classes", action="store_true",
                   help="no modo YOLO, detecta as 80 classes do COCO e não só pessoas")
    p.add_argument("--sem-rastreio", action="store_true", help="desativa o rastreamento (IDs)")
    p.add_argument("--capacidade", type=int, default=config.CAPACIDADE_SALA,
                   help="lotação máxima da sala (padrão: 5)")
    p.add_argument("--area-minima", type=int, default=config.AREA_MINIMA_CONTORNO,
                   help="modo OpenCV: área mínima do contorno em px² (padrão: 1500)")
    p.add_argument("--salvar", action="store_true", help="grava o vídeo processado em resultados/")
    p.add_argument("--sem-janela", action="store_true",
                   help="não abre janela (útil em servidores); implica --salvar")
    p.add_argument("--mostrar-mascara", action="store_true",
                   help="modo OpenCV: exibe também a máscara binária")
    return p.parse_args()


def criar_detector(args):
    """Instancia o detector escolhido."""
    if args.modo == "yolo":
        from src.detectores import DetectorYOLO
        classes = None if args.todas_classes else (config.CLASSE_PESSOA,)
        return DetectorYOLO(args.modelo, args.confianca, classes, rastrear=not args.sem_rastreio)
    from src.detectores import DetectorOpenCV
    return DetectorOpenCV(area_minima=args.area_minima)


def main():
    args = ler_argumentos()
    if args.sem_janela:
        args.salvar = True

    try:
        fonte = FonteVideo(args.fonte)
    except (FileNotFoundError, RuntimeError) as erro:
        print(f"[ERRO] {erro}")
        sys.exit(1)

    detector = criar_detector(args)
    est = Estatisticas(capacidade=args.capacidade)
    gravador = None
    pausado = False
    frame_saida = None

    print(f"[INFO] Modo {args.modo.upper()} | fonte: {args.fonte} | capacidade: {args.capacidade}")

    try:
        for frame in fonte.frames():
            # 1) Pré-processamento básico: redimensiona para acelerar
            frame = redimensionar(frame)

            # 2) Detecção (YOLO ou OpenCV)
            deteccoes = detector.detectar(frame)

            # 3) Métricas
            est.atualizar(deteccoes)

            # 4) Desenho dos resultados
            frame_saida = desenhar_deteccoes(frame, deteccoes)
            frame_saida = desenhar_painel(frame_saida, est, args.modo)

            # 5) Gravação opcional do vídeo processado
            if args.salvar and not fonte.eh_imagem:
                if gravador is None:
                    h, w = frame_saida.shape[:2]
                    caminho = os.path.join(config.PASTA_RESULTADOS, "saida_processada.mp4")
                    gravador = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"mp4v"), fonte.fps, (w, h))
                gravador.write(frame_saida)

            # 6) Exibição e teclado
            if not args.sem_janela:
                cv2.imshow(JANELA, frame_saida)
                if args.mostrar_mascara and getattr(detector, "ultima_mascara", None) is not None:
                    cv2.imshow("Mascara (OpenCV)", detector.ultima_mascara)

                espera = 0 if fonte.eh_imagem else 1   # imagem: espera uma tecla
                tecla = cv2.waitKey(espera) & 0xFF
                while pausado and tecla not in (ord("p"), ord("q")):
                    tecla = cv2.waitKey(50) & 0xFF
                if tecla == ord("p"):
                    pausado = not pausado
                elif tecla == ord("s"):
                    nome = os.path.join(config.PASTA_RESULTADOS, f"frame_{est.frames}.jpg")
                    cv2.imwrite(nome, frame_saida)
                    print(f"[INFO] Frame salvo em {nome}")
                elif tecla == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuário.")
    finally:
        # Para imagem estática, salva sempre o resultado anotado
        if fonte.eh_imagem and frame_saida is not None:
            caminho = os.path.join(config.PASTA_RESULTADOS, "imagem_processada.jpg")
            cv2.imwrite(caminho, frame_saida)
            print(f"[INFO] Imagem anotada salva em {caminho}")
        fonte.liberar()
        if gravador is not None:
            gravador.release()
        cv2.destroyAllWindows()
        print(est.finalizar())


if __name__ == "__main__":
    main()
