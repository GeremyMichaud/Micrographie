import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft2, fftshift
import os
import glob
import cv2
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
    threshold = 0.02 * np.max(spectral_density)
    regions_of_interest = (spectral_density > threshold).astype(np.uint8)

    # Appliquer une opération de fermeture pour regrouper les régions collées
    kernel = np.ones((5, 5), np.uint8)
    regions_of_interest_closed = cv2.morphologyEx(regions_of_interest, cv2.MORPH_CLOSE, kernel)

    # Trouver les contours des régions d'intérêt fermées
    contours, _ = cv2.findContours(regions_of_interest_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    spectral_density_colored = np.log(spectral_density)
    cv2.drawContours(spectral_density_colored, contours, -1, (0, 255, 0), 2)

    # Afficher l'image avec les contours des régions d'intérêt
    #plt.imshow(spectral_density_colored)
    #plt.show()

def process_image(image, tresh_factor=0.02):
    """
    Processus de traitement d'une image pour détecter les contours dans la densité spectrale.
    
    Args:
        image: np.ndarray, l'image en niveaux de gris
        tresh_factor: float, facteur de seuillage pour déterminer le seuil de binarisation
        
    Returns:
        list, une liste de contours des régions d'intérêt fermées
    """
    # Calcul de la densité spectrale
    spectrum = calculate_spectral_density(image)

    # Calcul du seuil
    threshold = tresh_factor * np.max(spectrum)

    # Binarisation de l'image
    regions_of_interest = (spectrum > threshold).astype(np.uint8)

    # Appliquer une opération de fermeture pour regrouper les régions collées
    kernel = np.ones((5, 5), np.uint8)
    regions_of_interest_closed = cv2.morphologyEx(regions_of_interest, cv2.MORPH_CLOSE, kernel)

    # Trouver les contours des régions d'intérêt fermées
    contours, _ = cv2.findContours(regions_of_interest_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def draw_contours(image, factor=0.02):
    cnt = process_image(image, factor)
    spectrum = calculate_spectral_density(image)
    rescaled_spectrum = np.log(spectrum)
    cv2.drawContours(rescaled_spectrum, cnt, -1, (0, 0, 0), 16)
    plt.imshow(rescaled_spectrum)
    plt.show()

for image in images:
    draw_contours(image)

