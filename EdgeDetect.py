import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from PIL import Image
from scipy.signal import convolve2d

img = Image.open('lena.jpg')
gray = np.mean(img, axis = 2)

Hx = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1])) # Sorbel Filter
Hy = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1])) # Prewitt filter

SGx = convolve2d(gray, Hx)
SGy = convolve2d(gray, Hx)

PGx = convolve2d(gray, Hy)
PGy = convolve2d(gray, Hy)

Sobel_out = np.sqrt( np.square(SGx) + np.square(SGy) )
Prewitt_out = np.sqrt( np.square(PGx) + np.square(PGy) )

plt.subplot(1,2,1)
plt.title('Sobel')
plt.imshow(Sobel_out, cmap = 'gray')
plt.subplot(1,2,2)
plt.title('Prewitt')
plt.imshow(Prewitt_out, cmap = 'gray')
plt.show()