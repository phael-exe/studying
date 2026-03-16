from __future__ import print_function
import cv2 as cv
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')
images = glob.glob(os.path.join(img_dir, '*.*'))

if not images:
    print('Nenhuma imagem encontrada na pasta:', img_dir)
    exit(0)

colors = ('b', 'g', 'r')
color_names = ('Azul', 'Verde', 'Vermelho')

for img_path in sorted(images):
    src = cv.imread(img_path)

    if src is None:
        print('Não foi possível abrir a imagem:', img_path)
        continue

    # Equalizar cada canal separadamente
    channels = cv.split(src)
    eq_channels = [cv.equalizeHist(ch) for ch in channels]
    dst = cv.merge(eq_channels)

    filename = os.path.basename(img_path)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f'Equalização de Histograma RGB - {filename}', fontsize=14, fontweight='bold')

    # Linha de cima: imagem original + histogramas originais por canal
    axes[0, 0].imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    for i, (col, name) in enumerate(zip(colors, color_names)):
        hist = cv.calcHist([src], [i], None, [256], [0, 256])
        axes[0, 1].plot(hist, color=col, label=name, alpha=0.8)
    axes[0, 1].set_title('Histograma Original')
    axes[0, 1].set_xlabel('Intensidade')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].legend()

    # Histogramas separados por canal (original)
    for i, (col, name) in enumerate(zip(colors, color_names)):
        hist = cv.calcHist([src], [i], None, [256], [0, 256])
        axes[0, 2].fill_between(range(256), hist.ravel(), color=col, alpha=0.3)
    axes[0, 2].set_title('Canais Originais (preenchido)')
    axes[0, 2].set_xlabel('Intensidade')
    axes[0, 2].set_ylabel('Frequência')

    # Linha de baixo: imagem equalizada + histogramas equalizados por canal
    axes[1, 0].imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    axes[1, 0].set_title('Equalizada (RGB)')
    axes[1, 0].axis('off')

    for i, (col, name) in enumerate(zip(colors, color_names)):
        hist = cv.calcHist([dst], [i], None, [256], [0, 256])
        axes[1, 1].plot(hist, color=col, label=name, alpha=0.8)
    axes[1, 1].set_title('Histograma Equalizado')
    axes[1, 1].set_xlabel('Intensidade')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].legend()

    for i, (col, name) in enumerate(zip(colors, color_names)):
        hist = cv.calcHist([dst], [i], None, [256], [0, 256])
        axes[1, 2].fill_between(range(256), hist.ravel(), color=col, alpha=0.3)
    axes[1, 2].set_title('Canais Equalizados (preenchido)')
    axes[1, 2].set_xlabel('Intensidade')
    axes[1, 2].set_ylabel('Frequência')

    plt.tight_layout()
    print(f'Mostrando: {filename}')
    plt.show()
