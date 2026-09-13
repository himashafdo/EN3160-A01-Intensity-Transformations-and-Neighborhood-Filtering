import numpy as np
import matplotlib.pyplot as plt
import cv2 

path = "images/q9.jpeg"
im9_src = cv2.imread(path)
im9 = cv2.cvtColor(im9_src, cv2.COLOR_BGR2RGB)


mask = np.zeros(im9.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

h, w = im9.shape[:2]
rect = (int(w*0.05), int(h*0.05),int(w*0.95),int(h*0.95)) #manually adjusted to get the flowers 
cv2.grabCut(im9, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')


#---(b)---
blurred = cv2.GaussianBlur(im9_src, (35, 35), 0)
blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

mask_3ch = mask_binary[:, :, np.newaxis]
enhanced = im9 * mask_3ch + blurred_rgb * (1 - mask_3ch)
enhanced = enhanced.astype(np.uint8)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(im9); ax[0].set_title('Original'); ax[0].axis('off')
ax[1].imshow(enhanced); ax[1].set_title('Background Blurred'); ax[1].axis('off')
plt.tight_layout()
plt.show()