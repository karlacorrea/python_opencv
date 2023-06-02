import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_gabor_filter(image, ksize, sigma, theta, lambd, gamma):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_gray = gray / 255.0

    gabor_kernel = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma)

    filtered_image = cv2.filter2D(normalized_gray, cv2.CV_32F, gabor_kernel)
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return filtered_image

image_path = r'C:\Users\mypc\Documents\python\001.jpeg'

# Verifique se o arquivo de imagem existe
try:
    with open(image_path, 'rb') as f:
        pass
except IOError:
    print(f"Arquivo de imagem não encontrado: {image_path}")
    exit()

image = cv2.imread(image_path)

if image is None:
    print(f"Erro ao carregar a imagem: {image_path}")
    exit()

ksize = 31
sigma = 4.0
theta = np.pi / 4
lambd = 10.0
gamma = 0.5

filtered_image = apply_gabor_filter(image, ksize, sigma, theta, lambd, gamma)

plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Imagem Original')
plt.subplot(1, 2, 2), plt.imshow(filtered_image, cmap='gray'), plt.title('Imagem Filtrada')
plt.show()
