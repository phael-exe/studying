# Harris Corner Detection — Explicação do Código

## Bibliotecas utilizadas

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

| Biblioteca | Função |
|---|---|
| `cv2` | OpenCV — visão computacional (leitura, conversão, detecção) |
| `numpy` | Manipulação de arrays numéricos |
| `matplotlib.pyplot` | Visualização das imagens |

---

## Passo 1 — Carregar e exibir a imagem original

```python
image = cv2.imread('/home/.../gengarjdm.jpg')
```

| Parâmetro | Descrição |
|---|---|
| `filename` | Caminho completo da imagem |
| retorno | Array NumPy no formato **BGR** (não RGB!) |

```python
image_copy = np.copy(image)
```

Cria uma **cópia independente** para não modificar o array original — boa prática essencial em visão computacional.

```python
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
```

| Parâmetro | Descrição |
|---|---|
| `src` | Imagem de entrada |
| `code` | `COLOR_BGR2RGB` — inverte a ordem dos canais. OpenCV lê em **BGR**, mas Matplotlib exibe em **RGB**. Sem isso, as cores ficam erradas. |

```python
plt.imshow(image_copy)
plt.axis('off')   # remove os eixos numéricos
plt.show()
```

---

## Passo 2 — Converter para escala de cinza

```python
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
```

Harris trabalha sobre **intensidade de pixel** (1 canal), não cor. A conversão calcula uma média ponderada dos canais RGB, resultando em 1 valor de luminância por pixel.

```python
gray = np.float32(gray)
```

O `cornerHarris` **exige** que a imagem seja `float32` (não `uint8`). Isso permite calcular derivadas com precisão decimal nas etapas seguintes.

---

## Passo 3 — Aplicar o Detector de Harris

```python
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
```

| Parâmetro | Nome | Valor usado | O que faz |
|---|---|---|---|
| `src` | imagem | `gray` | Imagem float32 em escala de cinza |
| `blockSize` | tamanho da janela | `2` | Tamanho da **vizinhança** analisada ao redor de cada pixel para montar a matriz de estrutura M |
| `ksize` | kernel Sobel | `3` | Tamanho do filtro Sobel para calcular os **gradientes** Ix e Iy (deve ser ímpar: 3, 5, 7...) |
| `k` | parâmetro Harris | `0.04` | Constante de sensibilidade da fórmula de resposta R. Valores típicos: **0.04 – 0.06** |
| retorno `dst` | mapa de resposta | — | Imagem onde cada pixel recebe um **score R** |

### A fórmula por trás do algoritmo

O Harris calcula, para cada pixel, a matriz de estrutura **M** usando os gradientes locais:

```
M = Σ [ Ix²    Ix·Iy ]
      [ Ix·Iy  Iy²   ]
```

A partir de M, calcula o **score de resposta R**:

```
R = det(M) - k · trace(M)²
```

| Resultado de R | Interpretação |
|---|---|
| **R >> 0** | ✅ Canto detectado |
| **R << 0** | Borda |
| **R ≈ 0** | Região plana (sem textura) |

### Efeito dos parâmetros

| Parâmetro | Valor menor | Valor maior |
|---|---|---|
| `blockSize` | Mais sensível a detalhes finos | Janela maior, mais robusto a ruído |
| `ksize` | Gradiente mais local | Gradiente mais suavizado |
| `k` | Detecta **mais** cantos (mais falsos positivos) | Detecta **menos** cantos (mais conservador) |

---

## Passo 4 — Dilatar para realçar os cantos

```python
dst = cv2.dilate(dst, None)
```

| Parâmetro | Valor | O que faz |
|---|---|---|
| `src` | `dst` | Mapa de resposta Harris |
| `kernel` | `None` | Usa kernel padrão 3×3 |

A **dilatação morfológica** faz cada pixel assumir o valor máximo da sua vizinhança. Isso **expande os picos de R**, tornando os cantos detectados mais visíveis na visualização.

---

## Passo 5 — Visualizar o mapa de cantos

```python
plt.imshow(dst, cmap='gray')
```

| Parâmetro | Valor | O que faz |
|---|---|---|
| `X` | `dst` | Mapa de resposta R após dilatação |
| `cmap` | `'gray'` | Colormap em escala de cinza — pixels **brancos** = alto R = **cantos detectados** |

> ⚠️ Atenção: `cmap='gray'` é uma **string**, não a variável `gray`.  
> Passar `cmap=gray` (sem aspas) causaria um `TypeError` pois o Matplotlib tentaria usar o numpy array como nome de colormap.

---

## Fluxo completo

```
Imagem (BGR)
    │
    ▼ cvtColor(BGR → RGB)
Imagem RGB          ← exibida com plt.imshow()
    │
    ▼ cvtColor(RGB → GRAY)
Imagem Cinza uint8
    │
    ▼ np.float32()
Imagem Cinza float32
    │
    ▼ cornerHarris(blockSize=2, ksize=3, k=0.04)
Mapa de Resposta R (dst)
    │
    ▼ dilate(kernel=None)
Mapa Dilatado
    │
    ▼ imshow(cmap='gray')
Visualização dos Cantos Detectados
```

---

## Dicas de experimento

- Varie o `k` entre `0.02` e `0.08` e observe a sensibilidade
- Aumente o `blockSize` para `4` ou `6` para janelas maiores
- Após a detecção, você pode marcar os cantos na imagem original com:

```python
image_copy[dst > 0.01 * dst.max()] = [255, 0, 0]  # pinta de vermelho
plt.imshow(image_copy)
```
