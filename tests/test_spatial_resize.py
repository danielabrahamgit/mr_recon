import torch

import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from mr_sim.phantoms import shepp_logan
from mr_recon.spatial import spatial_resize, spatial_resize_poly
from mr_recon.utils import np_to_torch

im_size = (256, 256)
low_res = (64, 64)
img = shepp_logan().img(im_size).abs()

img_rs = spatial_resize(img, low_res, method='bilinear')
img_back = spatial_resize(img_rs, im_size, method='bilinear')
# img_rs = spatial_resize_poly(img, low_res, order=1)
# img_back = spatial_resize_poly(img_rs, im_size, order=1)

plt.figure()
plt.imshow(img_rs)
plt.figure()
plt.imshow(img_back)
plt.show()