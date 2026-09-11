import cv2
import numpy as np
import matplotlib.pyplot as plt

im3 = cv2.imread("images/q3.jpeg")

lab_im = cv2.cvtColor(im3, cv2.COLOR_BGR2LAB)

l_channel, a_channel, b_channel = cv2.split(lab_im)

gamma = 0.6

normalized_l = l_channel / 255.0
corrected_l = np.power(normalized_l, gamma) * 255.0
corrected_l = np.uint8(np.clip(corrected_l, 0, 255))

corrected_lab = cv2.merge((corrected_l, a_channel, b_channel))

corrected_im = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].imshow(cv2.cvtColor(im3, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(corrected_im, cv2.COLOR_BGR2RGB))
ax[1].set_title("Gamma Corrected Image")
ax[1].axis('off')

plt.show()