import numpy as np
import matplotlib.pyplot as plt
import cv2

path = "images/q6.jpeg"
im6_bgr = cv2.imread(path)
im6_hsv = cv2.cvtColor(im6_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(im6_hsv)

Threshold = 14 # choesn from the valey between dark and light pixels
mask = (s > Threshold).astype(np.uint8) * 255

foreground = cv2.bitwise_and(v,v, mask=mask)
hist_fg, _ = np.histogram(foreground[mask==255], bins=256, range=(0, 255))

cdf = np.cumsum(hist_fg)

L = 256
num_fg_pixels = np.sum(mask == 255)

#use the formula give to us to compute the histogram equalization
t = np.array([((L-1)/ num_fg_pixels) * cdf[k] for k in range(256)], dtype=np.uint8)

fg_equalized = np.zeros_like(v)

fg_equalized[mask == 255] = t[v[mask == 255]] #equalization only on foreground

background = cv2.bitwise_and(v, v, mask=cv2.bitwise_not(mask))
v_new = background + fg_equalized

img_hsv_new = cv2.merge([h, s, v_new])
im6_result = cv2.cvtColor(img_hsv_new, cv2.COLOR_HSV2BGR)

fig, ax = plt.subplots(2, 3, figsize=(18, 6))
ax[0, 0].imshow(h, cmap = 'gray'); ax[0, 0].set_title('Hue'); ax[0, 0].axis('off')
ax[0, 1].imshow(s, cmap = 'gray'); ax[0, 1].set_title('Saturation'); ax[0, 1].axis('off')
ax[0, 2].imshow(v, cmap = 'gray'); ax[0, 2].set_title('Value'); ax[0, 2].axis('off')

ax[1, 0].imshow(mask, cmap = 'gray'); ax[1, 0].set_title('mask'); ax[1, 0].axis('off')
ax[1, 1].imshow(cv2.cvtColor(im6_bgr, cv2.COLOR_BGR2RGB)); ax[1, 1].set_title('Original Image'); ax[1, 1].axis('off')
ax[1, 2].imshow(cv2.cvtColor(im6_result, cv2.COLOR_BGR2RGB)); ax[1, 2].set_title('Resulting Image'); ax[1, 2].axis('off')

plt.show()