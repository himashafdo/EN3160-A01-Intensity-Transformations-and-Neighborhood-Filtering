import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 

path = "images/q4.jpeg"
im4 = cv2.imread(path)

def vibrance_transform(x, a, sigma=70):

    x = x.astype(float)
    boost = a * 128 * np.exp(-((x - 128) ** 2) / (2 * sigma ** 2))
    out = np.minimum(x + boost, 255)

    return out.astype(np.uint8)

#---(a)---
im4_hsv = cv2.cvtColor(im4, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(im4_hsv)

#---(b) & (c)---
a = 0.65
s_new = vibrance_transform(s, a)

#---(d)---
im4_hsv_new = cv2.merge([h, s_new, v])
im4_bgr_new = cv2.cvtColor(im4_hsv_new, cv2.COLOR_HSV2BGR)

#---(e)---
xs = np.arange(0, 256)
lut2 = vibrance_transform(xs, a)

fig, ax = plt.subplots(1, 3, figsize=(16, 6))

ax[0].imshow(cv2.cvtColor(im4, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Image"); ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(im4_bgr_new, cv2.COLOR_BGR2RGB))
ax[1].set_title("Vibrance Adjusted Image"); ax[1].axis('off')

ax[2].plot(xs, lut2, color='navy', linewidth=2)
ax[2].plot([0,255], [0,255], '--' , color='gray', linewidth=1)
ax[2].set_xlabel('Input Saturation')
ax[2].set_ylabel('Output Saturation'); ax[2].set_title('Saturation Transform')
ax[2].set_xlim(0, 255); ax[2].set_ylim(0, 255); ax[2].grid()

plt.show()
