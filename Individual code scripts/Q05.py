import numpy as np
import matplotlib.pyplot as plt
import cv2 

def histogram_equalization(im):

    hist, bins = np.histogram(im.flatten(), bins=256, range=(0, 256))
    
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    total_pixels = im.size

    lut = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    out = lut[im]
    return out, lut

path = "images/shells.tif"
im5 = cv2.imread(path)

out5, lut3 = histogram_equalization(im5)

fig, ax = plt.subplots(2, 2, figsize=(16, 6))

ax[0, 0].imshow(im5, cmap='gray')
ax[0, 0].set_title("Original Image"); ax[0, 0].axis('off')

ax[0, 1].imshow(out5, cmap='gray')
ax[0, 1].set_title("Histogram Equalized Image"); ax[0, 1].axis('off')

ax[1, 0].hist(im5.ravel(), bins=256, range=(0,255), color='navy')
ax[1, 0].set_title("Histogram of Original Image")
ax[1, 0].set_xlabel("Intensity Value"); ax[1, 0].set_ylabel("Pixel Count")

ax[1, 1].hist(out5.ravel(), bins=256, range=(0,255), color='navy')
ax[1, 1].set_title("Histogram of Equalized Image")
ax[1, 1].set_xlabel("Intensity Value"); ax[1, 1].set_ylabel("Pixel Count")

plt.show()