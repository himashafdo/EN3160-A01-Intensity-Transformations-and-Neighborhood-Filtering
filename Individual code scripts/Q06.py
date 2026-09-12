import numpy as np
import matplotlib.pyplot as plt
import cv2

path = "images/q6.jpeg"
im6_bgr = cv2.imread(path)
im6_hsv = cv2.cvtColor(im6_bgr, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(im6_hsv)

# fig, ax = plt.subplots(1, 3, figsize=(15,5))
# ax[0].imshow(h, cmap='gray'); ax[0].set_title('Hue'); ax[0].axis('off')
# ax[1].imshow(s, cmap='gray'); ax[1].set_title('Saturation'); ax[1].axis('off')
# ax[2].imshow(v, cmap='gray'); ax[2].set_title('Value'); ax[2].axis('off')

#plt.show() #used to understand which plane to use 

#plt.hist(s.ravel(), bins=256, range=(0, 255))
#plt.title("Histogram of saturation plane")
#plt.show() # to understand the threshold value

T = 14 # choesn from the valey between dark and light pixels
mask = (s > T).astype(np.uint8) * 255

foreground = cv2.bitwise_and(v,v, mask=mask)
hist_fg, _ = np.histogram(foreground[mask==255], bins=256, range=(0, 255))

cdf = np.cumsum(hist_fg)

L = 256
num_fg_pixels = np.sum(mask == 255)

t = np.array([(L-1)/ num_fg_pixels * cdf[k] for k in range(256)], dtype=np.uint8)


fg_equalized = np.zeros_like(v)
fg_equalized[mask == 255] = t[v[mask == 255]]

background = cv2.bitwise_and(v, v, mask=cv2.bitwise_not(mask))
v_new = background + fg_equalized

img_hsv_new = cv2.merge([h, s, v_new])
img_result = cv2.cvtColor(img_hsv_new, cv2.COLOR_HSV2BGR)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(cv2.cvtColor(im6_bgr, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
ax[1].set_title('Foreground Histogram Equalized')
ax[1].axis('off')
plt.tight_layout()
plt.show()