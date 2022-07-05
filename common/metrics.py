''' metrics '''
import math
import numpy as np


def calc_psnr(img, img_target):
    ''' calc_psnr '''
    img = np.clip(np.round(img), 0, 255).astype(np.float32)
    img_target = np.clip(np.round(img_target), 0, 255).astype(np.float32)
    mse = np.mean((img - img_target) ** 2)
    if mse == 0:
        return 100
    return 20. * math.log10(255. / math.sqrt(mse))
