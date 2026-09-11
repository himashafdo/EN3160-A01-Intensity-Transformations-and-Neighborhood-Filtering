import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path = "images/q2.jpeg"
im = np.array(Image.open(path).convert('L'))    

plt.hist(im.ravel(), bins=256, range=(0,255))
plt.show() #histogram of the original image'

