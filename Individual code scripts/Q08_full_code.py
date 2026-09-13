import cv2
import numpy as np
import matplotlib.pyplot as plt

def zoom_image(im, s, method):
    """"
    Defined function to output the zoomed image with the factor of s using either 
    bilinear or nearest neighbour method.
    """

    im = im.astype(np.float64)
    H, W = im.shape
    new_H, new_W = int(round(H*s)), int(round(W*s))

    out = np.zeros((new_H, new_W), dtype = np.float64)

    for i in range(new_H):
        for j in range(new_W):
            y_src = i / s 
            x_src = j / s

            if method == 'nearest':
                y0 = min(int(round(y_src)), H-1)
                x0 = min(int(round(x_src)), W-1)
                out[i, j] = im[y0, x0]

            elif method == 'bilinear':
                x0 = int(np.floor(x_src))
                y0 = int(np.floor(y_src))
                x1 = min(x0 + 1, W-1)
                y1 = min(y0 + 1, H-1)
                x0 = min(x0, W-1)
                y0 = min(y0 , H-1)

                dx = x_src-x0
                dy = y_src-y0

                top = im[y0, x0] * (1 - dx) + im[y0, x1] * dx
                bottom = im[y1, x0] * (1 - dx) + im[y1, x1] * dx
                out[i, j] = top * (1 - dy) + bottom * dy

    return out

def zoom_image_color(im, s, method):
    """
    Zooms a color image by applying zoom_image independently to each color channel (B, G, R).
    """
    channels = cv2.split(im)  
    zoomed_channels = [zoom_image(ch, s, method) for ch in channels]
    return cv2.merge(zoomed_channels)

def normalized_ssd(im1, im2):
    """
    Here I defined another function to calculate the normalized ssd.
    """
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    ssd = np.sum((im1 - im2) ** 2)
    normalized = ssd / im1.size
    return normalized

s = 4

#-- for image 01 ---
small_im1 = cv2.imread("images/im01small.png")
large_im1 = cv2.imread("images/im01.png")
zoomed_nn1 = zoom_image_color(small_im1, s, method='nearest')
zoomed_bilinear1 = zoom_image_color(small_im1, s, method='bilinear')
#print(large_im1.shape == zoomed_nn1.shape)
ssd_nn1 = normalized_ssd(zoomed_nn1, large_im1)
ssd_bilinear1 = normalized_ssd(zoomed_bilinear1, large_im1)

#-- for image 02 ---
small_im2 = cv2.imread("images/im02small.png")
large_im2 = cv2.imread("images/im02.png")
zoomed_nn2 = zoom_image_color(small_im2, s, method='nearest')
zoomed_bilinear2 = zoom_image_color(small_im2, s, method='bilinear')
#print(large_im2.shape == zoomed_nn2.shape) #
ssd_nn2 = normalized_ssd(zoomed_nn2, large_im2)
ssd_bilinear2 = normalized_ssd(zoomed_bilinear2, large_im2)

fig, ax = plt.subplots(2, 3, figsize=(18, 6))
ax[0, 0].imshow(cv2.cvtColor(np.clip(small_im1, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)); ax[0, 0].set_title('Small (input)'); ax[0, 0].axis('off')
ax[0, 1].imshow(cv2.cvtColor(np.clip(zoomed_nn1, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)); ax[0, 1].set_title(f'Zoomed 4x - Nearest\nSSD={ssd_nn1:.2f}'); ax[0, 1].axis('off')
ax[0, 2].imshow(cv2.cvtColor(np.clip(zoomed_bilinear1, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)); ax[0, 2].set_title(f'Zoomed 4x - Bilinear\nSSD={ssd_bilinear1:.2f}'); ax[0 ,2].axis('off')

ax[1, 0].imshow(cv2.cvtColor(np.clip(small_im2, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)); ax[1, 0].set_title('Small (input)'); ax[1, 0].axis('off')
ax[1, 1].imshow(cv2.cvtColor(np.clip(zoomed_nn2, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)); ax[1, 1].set_title(f'Zoomed 4x - Nearest\nSSD={ssd_nn2:.2f}'); ax[1, 1].axis('off')
ax[1, 2].imshow(cv2.cvtColor(np.clip(zoomed_bilinear2, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)); ax[1, 2].set_title(f'Zoomed 4x - Bilinear\nSSD={ssd_bilinear2:.2f}'); ax[1 ,2].axis('off')
plt.show()


