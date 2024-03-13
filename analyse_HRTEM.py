import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft2, ifft2, fftshift
import os
import glob
import imageio.v2 as imageio
import seaborn as sns
palette = sns.color_palette("colorblind")


images_path = glob.glob("images/*.tif")

images = [imageio.imread(chemin_image) for chemin_image in images_path]


def calculate_spectral_density(image):
    """
    Calcule la densité spectrale d'une image.
    
    Args:
        image: np.ndarray, l'image en niveaux de gris
        
    Returns:
        np.ndarray, la densité spectrale de l'image
    """
    spectrum = fftshift(fft2(image))
    spectral_density = np.abs(spectrum)
    
    return spectral_density

for image in images:
    spectral_density = calculate_spectral_density(image)
    plt.imshow(np.log(spectral_density), cmap='magma')
    plt.show()