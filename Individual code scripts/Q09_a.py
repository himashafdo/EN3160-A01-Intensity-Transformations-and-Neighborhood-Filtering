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

foreground = im9 * mask_binary[:, :, np.newaxis]
background = im9 * (1 - mask_binary[:, :, np.newaxis])

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].imshow(mask_binary, cmap='gray'); ax[0].set_title('Segmentation Mask'); ax[0].axis('off')
ax[1].imshow(foreground); ax[1].set_title('Foreground'); ax[1].axis('off')
ax[2].imshow(background); ax[2].set_title('Background'); ax[2].axis('off')
plt.show()