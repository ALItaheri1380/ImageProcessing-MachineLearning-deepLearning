import sys
from tqdm import trange
import numpy as np
import cv2
from imageio.v3 import imread, imwrite

def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
    return (rgb @ coeffs).astype(rgb.dtype)

def normalize(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

def calc_gmap(gray):
    gray = gray.astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.sqrt(grad_x**2 + grad_y**2)
    gmap = normalize(energy)
    return gmap

def calc_energy(img):
    global smap, dmap

    gray = img
    if np.ndim(img) != 2:
        gray = rgb2gray(img)
    gmap = calc_gmap(gray)
    emap = smap.astype(np.double) * 0.4 + gmap + 1.2 * dmap
    return emap

def carve_column(img):
    global smap, dmap

    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    smap = smap[mask].reshape((r, c-1))
    dmap = dmap[mask].reshape((r, c-1))
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.astype(np.uint32)
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def crop(img, scale_c):
    _, c, _ = img.shape
    new_c = int(scale_c * c)

    for _ in trange(c - new_c):
        img = carve_column(img)
        imwrite("processing.png", img)

    return img

dataset_path = './Samples dataset/'
def main():
    global dataset_path, smap, dmap

    if len(sys.argv) != 3:
        print('usage: seam_carving.py <scale> <image_name>', file=sys.stderr)
        sys.exit(1)

    scale = float(sys.argv[1])
    image = sys.argv[2]
    out_filename = f'{image}_{scale}.png'

    dataset_path = dataset_path + image + '/'
    smap = imread(dataset_path + image + '_SMap.png')
    smap = normalize(smap)
    dmap = imread(dataset_path + image + '_DMap.png')
    dmap = normalize(dmap)

    img = imread(dataset_path + image + '.png')

    out = crop(img, scale)
    
    imwrite(out_filename, out)

if __name__ == '__main__':
    main()