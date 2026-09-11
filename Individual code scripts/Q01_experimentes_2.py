import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def intensity_transform(im, breakpoints):
    """
        Here I defined the intensity transformation function to precompute values for 
        each pixel value in the range [0, 255] and store them in a lookup table, 
        since it is much faster than computing value for the pixel value each time.
            
    """
    out = im.copy().astype(float)

    for i in range(len(breakpoints) - 1):

        x1, y1 = breakpoints[i]
        x2, y2 = breakpoints[i + 1]

        if x2 == x1:
            continue # skips vertical lines as no need of computing slope there.
        else:
            slope = (y2 - y1) / (x2 - x1)
            mask = (im >= x1) & (im <= x2)
            out[mask] = y1 + slope * (im[mask] - x1)

    return out

path = "images/emma_gray.jpg"
im = np.array(Image.open(path).convert('L'))

#plt.hist(im.ravel(), bins=256, range=(0,255))
#plt.show() #histogram of the original image

breakpoints = [
    [0, 0],
    [60, 50],
    [100, 130],
    [200, 210],
    [255, 255],
]

out = intensity_transform(im, breakpoints)

xs = np.arange(0, 256)
lut = intensity_transform(xs, breakpoints)

bp = np.array(breakpoints)

fig, ax = plt.subplots(1, 3, figsize=(16, 6))

ax[0].plot(xs, lut, color='navy', linewidth=2)
ax[0].scatter(bp[:, 0], bp[:, 1], color='red', s=15, zorder=5)
ax[0].set_xlabel('Input Intensity')
ax[0].set_ylabel('Output Intensity')
ax[0].set_title('Intensity Transformation')
ax[0].set_xlim(0, 255); ax[0].set_ylim(0, 255)
ax[0].set_xticks([0, 50, 100, 150, 200, 255]); ax[0].set_yticks([0, 50, 100, 150, 200, 255])
ax[0].grid()

ax[1].imshow(im, cmap='gray')
ax[1].set_title('Original Image')
ax[1].axis('off')

ax[2].imshow(out.astype(np.uint8), cmap='gray')
ax[2].set_title('Transformed Image')
ax[2].axis('off')

plt.savefig("results/experimented_02.png", dpi=150)
plt.show()

