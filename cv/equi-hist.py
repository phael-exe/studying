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

for img_path in sorted(images):
    src = cv.imread(img_path)

    if src is None:
        print('Não foi possível abrir a imagem:', img_path)
        continue

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)

    filename = os.path.basename(img_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Equalização de Histograma - {filename}', fontsize=14, fontweight='bold')

    # Imagem original
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Histograma original
    axes[0, 1].hist(gray.ravel(), bins=256, range=(0, 256), color='steelblue', alpha=0.8)
    axes[0, 1].set_title('Histograma Original')
    axes[0, 1].set_xlabel('Intensidade')
    axes[0, 1].set_ylabel('Frequência')

    # Imagem equalizada
    axes[1, 0].imshow(dst, cmap='gray')
    axes[1, 0].set_title('Equalizada')
    axes[1, 0].axis('off')

    # Histograma equalizado
    axes[1, 1].hist(dst.ravel(), bins=256, range=(0, 256), color='coral', alpha=0.8)
    axes[1, 1].set_title('Histograma Equalizado')
    axes[1, 1].set_xlabel('Intensidade')
    axes[1, 1].set_ylabel('Frequência')

    plt.tight_layout()
    print(f'Mostrando: {filename}')
    plt.show()