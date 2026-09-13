import numpy as np
import matplotlib.pyplot as plt
import cv2 

path = "images/q7.jpeg"
im7 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

Gx_kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
Gy_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)

#---(a)---
gx_a = cv2.filter2D(im7.astype(np.float32), -1, Gx_kernel)
gy_a = cv2.filter2D(im7.astype(np.float32), -1, Gy_kernel)
grad_mag_a = np.sqrt(gx_a**2 + gy_a**2)
grad_mag_a = np.clip(grad_mag_a, 0, 255).astype(np.uint8)

#---(b)---
def defined_sobelfilter(im, kernel):

    im = im.astype(np.float64)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw// 2

    padded = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant') #zero paddint added

    out = np.zeros_like(im)
    H, W = im.shape

    for i in range(H):
        for j in range(W):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)

    return out

gx_b = defined_sobelfilter(im7, Gx_kernel)
gy_b = defined_sobelfilter(im7, Gy_kernel)
grad_mag_b = np.sqrt(gx_b**2 + gy_b**2)
grad_mag_b = np.clip(grad_mag_b, 0, 255).astype(np.uint8)

#---(c)---
def separable_sobel_x(im):
    im = im.astype(np.float64)
    
    c_1 = np.array([[1], [2], [1]], dtype=np.float64)   # (column vector)
    c_2 = np.array([[1, 0, -1]], dtype=np.float64)        # (row vector)
     
    c_1 = defined_sobelfilter(im, c_1)
    out = defined_sobelfilter(c_1, c_2)
    
    return out

gx_c = separable_sobel_x(im7)

def separable_sobel_y(im):
    im = im.astype(np.float64)

    c_3 = np.array([[1, 2, 1]], dtype=np.float64)        
    c_4 = np.array([[1], [0], [-1]], dtype=np.float64) 

    c_3 = defined_sobelfilter(im, c_3)
    out = defined_sobelfilter(c_3, c_4)
    return out

gy_c = separable_sobel_y(im7)

grad_mag_c = np.sqrt(gx_c**2 + gy_c**2)
grad_mag_c = np.clip(grad_mag_c, 0, 255).astype(np.uint8)

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(im7, cmap='gray'); ax[0].set_title('Original'); ax[0].axis('off')
ax[1].imshow(grad_mag_a, cmap='gray'); ax[1].set_title('(a) filter2D'); ax[1].axis('off')
ax[2].imshow(grad_mag_b, cmap='gray'); ax[2].set_title('(b) Defined function'); ax[2].axis('off')
ax[3].imshow(grad_mag_c, cmap='gray'); ax[3].set_title('(c) Separable Method'); ax[3].axis('off')
plt.show()