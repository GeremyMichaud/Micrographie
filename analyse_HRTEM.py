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
images_dic = {os.path.splitext(os.path.basename(chemin_image))[0]: imageio.imread(chemin_image) for chemin_image in images_path}

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

def process_image(image, tresh_factor):
    """
    Processus de traitement d'une image pour détecter les contours dans la densité spectrale.
    
    Args:
        image: np.ndarray, l'image en niveaux de gris
        tresh_factor: float, facteur de seuillage pour déterminer le seuil de binarisation
        
    Returns:
        list, une liste de contours des régions d'intérêt fermées
    """
    spectrum = calculate_spectral_density(image)
    threshold = tresh_factor * np.max(spectrum)
    regions_of_interest = (spectrum > threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    regions_of_interest_closed = cv2.morphologyEx(regions_of_interest, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(regions_of_interest_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def draw_contours(image, name, factor=0.02):
    """
    Dessine les contours des régions d'intérêt sur l'image de densité spectrale.
    
    Args:
        image: np.ndarray, l'image en niveaux de gris
        name: string,  nom du fichier à sauvegarder
        factor: float, facteur de seuillage pour déterminer le seuil de binarisation
    """
    cnt = process_image(image, factor)
    spectrum = calculate_spectral_density(image)
    rescaled_spectrum = cv2.normalize(np.log(spectrum), None, 0, 255, cv2.NORM_MINMAX)
    colored_spectrum = cv2.cvtColor(rescaled_spectrum.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for i, contour in enumerate(cnt):
        cv2.drawContours(colored_spectrum, [contour], -1, (0, 0, 255), 2)
        contour_center = contour.mean(axis=0).astype(int)[0]
        cv2.putText(colored_spectrum, str(i), tuple(contour_center), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    directory = os.path.join("output", "all_contours")
    if not os.path.exists(directory):
            os.makedirs(directory)

    cv2.imwrite(os.path.join(directory, name + ".png"), colored_spectrum)

def remove_contours(image, name, contours_to_remove, factor=0.02):
    """
    Remove specified contours from the image.
    
    Args:
        image: np.ndarray, the grayscale image
        name: str, the file name to save
        contours_to_remove: list, list of contour indices to remove
        factor: float, thresholding factor to determine the binarization threshold
    """
    cnt = process_image(image, factor)
    spectrum = calculate_spectral_density(image)
    rescaled_spectrum = cv2.normalize(np.log(spectrum), None, 0, 255, cv2.NORM_MINMAX)
    colored_spectrum = cv2.cvtColor(rescaled_spectrum.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Remove specified contours
    cnt = [contour for i, contour in enumerate(cnt) if i not in contours_to_remove]

    for i, contour in enumerate(cnt):
        cv2.drawContours(colored_spectrum, [contour], -1, (0, 0, 255), 2)
        contour_center = contour.mean(axis=0).astype(int)[0]
        cv2.putText(colored_spectrum, str(i), tuple(contour_center), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    directory = os.path.join("output", "contours_removed")
    if not os.path.exists(directory):
            os.makedirs(directory)

    cv2.imwrite(os.path.join(directory, name + ".png"), colored_spectrum)

    return cnt


image_5_name, image_6_name, image_7_name = images_dic.keys()
image_5, image_6, image_7 = images_dic.values()
draw_contours(image_5, image_5_name, factor=0.015)
draw_contours(image_6, image_6_name, factor=0.028)
draw_contours(image_7, image_7_name, factor=0.0182)

contour2remove_5 = [6, 8]
contour2remove_6 = [3, 12]
contour2remove_7 = [2, 3, 4, 6, 7, 9, 10, 13, 14, 15, 17, 18]
contours_5 = remove_contours(image_5, image_5_name, contour2remove_5, factor=0.015)
contours_6 = remove_contours(image_6, image_6_name, contour2remove_6, factor=0.028)
contours_7 = remove_contours(image_7, image_7_name, contour2remove_7, factor=0.0182)