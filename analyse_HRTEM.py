import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import os
import glob
import cv2
import imageio.v2 as imageio
import seaborn as sns
palette = sns.color_palette("colorblind")


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
    threshold = tresh_factor * np.max(image)
    regions_of_interest = (image > threshold).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    regions_of_interest_closed = cv2.morphologyEx(regions_of_interest, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(regions_of_interest_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def draw_all_contours(image, name, factor=0.02):
    """
    Dessine les contours des régions d'intérêt sur l'image de densité spectrale.
    
    Args:
        image: np.ndarray, l'image en niveaux de gris
        name: string,  nom du fichier à sauvegarder
        factor: float, facteur de seuillage pour déterminer le seuil de binarisation
    """
    cnt = process_image(image, factor)
    rescaled_spectrum = cv2.normalize(np.log(image), None, 0, 255, cv2.NORM_MINMAX)
    colored_spectrum = cv2.cvtColor(rescaled_spectrum.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for i, contour in enumerate(cnt):
        cv2.drawContours(colored_spectrum, [contour], -1, (0, 0, 255), 2)
        contour_center = contour.mean(axis=0).astype(int)[0]
        cv2.putText(colored_spectrum, str(i+1), tuple(contour_center), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    directory = os.path.join("output", "01_all_contours")
    if not os.path.exists(directory):
            os.makedirs(directory)

    cv2.imwrite(os.path.join(directory, name + ".png"), colored_spectrum)

def remove_contours(image, contours_to_remove, factor=0.02):
    """
    Remove specified contours from the list.
    
    Args:
        image: np.ndarray, the grayscale image
        contours_to_remove: list, list of contour indices to remove
        factor: float, thresholding factor to determine the binarization threshold
    """
    bad_cnt = process_image(image, factor)
    good_cnt = [contour for i, contour in enumerate(bad_cnt) if i not in contours_to_remove]

    return good_cnt

def draw_good_contours(image, name, contours):
    """
    Draw only good contours on an image and save it.

    Args:
        image (np.ndarray): Input grayscale image.
        name (str): Name of the output image file.
        contours (list): List of contours to draw on the image.
    """
    rescaled_spectrum = cv2.normalize(np.log(image), None, 0, 255, cv2.NORM_MINMAX)
    colored_spectrum = cv2.cvtColor(rescaled_spectrum.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for i, contour in enumerate(contours):
        cv2.drawContours(colored_spectrum, [contour], -1, (0, 0, 255), 2)
        contour_center = contour.mean(axis=0).astype(int)[0]
        cv2.putText(colored_spectrum, str(i+1), tuple(contour_center), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    directory = os.path.join("output", "02_contours_removed")
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(os.path.join(directory, name + ".png"), colored_spectrum)


def gaussian(x, a, x0, sigma):
    """
    Gaussian function.
    
    Args:
        x (array-like): Input data.
        a (float): Amplitude of the Gaussian curve.
        x0 (float): Mean of the Gaussian curve.
        sigma (float): Standard deviation of the Gaussian curve.
        
    Returns:
        array-like: Values of the Gaussian function evaluated at x.
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def find_contour_centroids(contours, image):
    """
    Find centroids of contours in an image.

    Args:
        contours (list): List of contours.
        image (np.ndarray): Grayscale image.

    Returns:
        tuple: A tuple containing two lists. The first list contains tuples of centroid coordinates (x, y) for each contour.
            The second list contains tuples of centroid uncertainties (x_uncertainty, y_uncertainty) for each contour.
    """
    centroids = []
    centroids_uncertainty = []
    for contour in contours:
        x_coords = contour[:, 0, 0]
        y_coords = contour[:, 0, 1]

        min_x_coord = np.min(x_coords)
        max_x_coord = np.max(x_coords)
        min_y_coord = np.min(y_coords)
        max_y_coord = np.max(y_coords)

        if len(x_coords) < 5 or len(y_coords) < 5:
            x_center, y_center = int(len(x_coords) / 2) , int(len(y_coords) / 2)
            x_center_pix, y_center_pix = min_x_coord + x_center, min_y_coord + y_center
            centroids.append((x_center_pix, y_center_pix))
            centroids_uncertainty.append((0, 0))
            continue

        spectrum = calculate_spectral_density(image)
        pixel_values = spectrum[min_y_coord:max_y_coord, min_x_coord:max_x_coord]

        x_mean_intensity = np.mean(pixel_values, axis=0)
        y_mean_intensity = np.mean(pixel_values, axis=1)

        x_data = np.arange(len(x_mean_intensity))
        popt_x, cov_x = curve_fit(gaussian, x_data, x_mean_intensity, p0=[np.max(x_mean_intensity), len(x_data) / 2, 10])
        y_data = np.arange(len(y_mean_intensity))
        popt_y, cov_y = curve_fit(gaussian, y_data, y_mean_intensity, p0=[np.max(y_mean_intensity), len(y_data) / 2, 10])

        x_center = int(popt_x[1])
        x_center_pix = min_x_coord + x_center
        x_center_uncertainty = np.sqrt(np.diag(cov_x)[1])
        y_center = int(popt_y[1])
        y_center_pix = min_y_coord + y_center
        y_center_uncertainty = np.sqrt(np.diag(cov_y)[1])

        centroids.append((x_center_pix, y_center_pix))
        centroids_uncertainty.append((x_center_uncertainty, y_center_uncertainty))

    return centroids, centroids_uncertainty

def find_pairs(image, centroids):
    """
    Trouve les paires de centroïdes symétriques.


    Args:
        image (np.ndarray): L'image.
        centroids (list): Liste des coordonnées des centroïdes à rechercher.

    Returns:
        list: Liste de paires de centroïdes symétriques.
    """
    pos_pairs = []
    centroids2match = centroids.copy()

    for c in centroids2match:
        y, x = c

        center_y = int(image.shape[0] / 2) - 10 <= y <= int(image.shape[0] / 2) + 10
        center_x = int(image.shape[1] / 2) - 10 <= x <= int(image.shape[1] / 2) + 10
        if center_y and center_x:
            continue

        y_prime = 2 * (int(image.shape[0] / 2) - y) + y
        x_prime = 2 * (int(image.shape[1] / 2) - x) + x

        for c_prime in centroids2match:
            y_prime_centroid, x_prime_centroid = c_prime
            if abs(y_prime_centroid - y_prime) <= 10 and abs(x_prime_centroid - x_prime) <= 10:
                pos_pairs.append((c, c_prime))
                centroids2match.remove(c_prime)
                break

    return pos_pairs

def draw_pairs(image, name, pairs):
    """
    Draw pairs of points.

    Args:
        image (np.ndarray): Input grayscale image.
        name (str): Name of the output image file.
        pairs (list): List of pairs of points to draw on the image.
    """
    rescaled_spectrum = cv2.normalize(np.log(image), None, 0, 255, cv2.NORM_MINMAX)
    colored_spectrum = cv2.cvtColor(rescaled_spectrum.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for i, (start_point, end_point) in enumerate(pairs):
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.line(colored_spectrum, start_point, end_point, color, 1, cv2.LINE_AA)
        for point in [start_point, end_point]:
            cv2.putText(colored_spectrum, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    out_dir = os.path.join("output", "03_pairs")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, name + ".png"), colored_spectrum)

def create_cos_fourier_spaces(image, pairs):
    """
    This function generates Fourier spaces representing the cosine frequency of interest.
    It places the signal at the origin and one of the points of each selected pair.

    Args:
        image (np.ndarray): The input image.
        pairs (list): A list of tuples representing pairs of points.

    Returns:
        tuple: A tuple containing two lists:
            - The first list contains Fourier spaces representing the cosine frequencies.
            - The second list contains the slopes of the lines connecting the pairs of points.
    """
    slopes = []
    fft_lines = []

    y_center = image.shape[0] // 2
    x_center = image.shape[1] // 2

    for (first_point, second_point) in pairs:
        y1, x1 = first_point
        y2, x2 = second_point

        fft_line = np.zeros(image.shape, dtype=np.int32)
        fft_line[y1][x1] = 1000
        fft_line[y_center][x_center] = 1000
        fft_lines.append(fft_line)
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)

    return fft_lines, slopes

def inverse_fourier_transform(fft_lines):
    """
    Compute the inverse Fourier transform for a list of FFT lines.

    Args:
        fft_lines (list): List of FFT lines.

    Returns:
        list: List of waves obtained from the inverse Fourier transform of each FFT line.
    """
    waves = [np.abs(ifft2(ifftshift(fft_line))) for fft_line in fft_lines]
    return waves


images_path = glob.glob("images/*.tif")
images_dic = {os.path.splitext(os.path.basename(chemin_image))[0]: imageio.imread(chemin_image) for chemin_image in images_path}

image_5_name, image_6_name, image_7_name = images_dic.keys()
image_5, image_6, image_7 = images_dic.values()
spectrum_5 = calculate_spectral_density(image_5)
spectrum_6 = calculate_spectral_density(image_6)
spectrum_7 = calculate_spectral_density(image_7)
#draw_all_contours(spectrum_5, image_5_name, factor=0.015)
#draw_all_contours(spectrum_6, image_6_name, factor=0.028)
#draw_all_contours(spectrum_7, image_7_name, factor=0.0182)

contour2remove_5 = [6, 8]
contour2remove_6 = [3, 12]
contour2remove_7 = [2, 3, 4, 6, 7, 9, 10, 13, 14, 15, 17, 18]
contours_5 = remove_contours(spectrum_5, contour2remove_5, factor=0.015)
contours_6 = remove_contours(spectrum_6, contour2remove_6, factor=0.028)
contours_7 = remove_contours(spectrum_7, contour2remove_7, factor=0.0182)
#draw_good_contours(spectrum_5, image_5_name, contours_5)
#draw_good_contours(spectrum_6, image_6_name, contours_6)
#draw_good_contours(spectrum_7, image_7_name, contours_7)

centroids_5, uncert_5 = find_contour_centroids(contours_5, image_5)
centroids_6, uncert_6 = find_contour_centroids(contours_6, image_6)
centroids_7, uncert_7 = find_contour_centroids(contours_7, image_7)

pos_pairs_5 = find_pairs(spectrum_5, centroids_5)
pos_pairs_6 = find_pairs(spectrum_6, centroids_6)
pos_pairs_7 = find_pairs(spectrum_7, centroids_7)
#draw_pairs(spectrum_5, image_5_name, pos_pairs_5)
#draw_pairs(spectrum_6, image_6_name, pos_pairs_6)
#draw_pairs(spectrum_7, image_7_name, pos_pairs_7)

fft_lines_5, slopes_5 = create_cos_fourier_spaces(spectrum_5, pos_pairs_5)
fft_lines_6, slopes_6 = create_cos_fourier_spaces(spectrum_6, pos_pairs_6)
fft_lines_7, slopes_7 = create_cos_fourier_spaces(spectrum_7, pos_pairs_7)

waves_5 = inverse_fourier_transform(fft_lines_5)
waves_6 = inverse_fourier_transform(fft_lines_6)
waves_7 = inverse_fourier_transform(fft_lines_7)

nb_plan = len(fft_lines_5)
rows = nb_plan//3
if nb_plan%3 != 0:
	rows += 1
fig, ax = plt.subplots(nrows=rows, ncols=3)
plt.subplots_adjust(wspace=0.01, hspace=0.01)
for i in range(3*rows):
    y = i//3
    x = i%3
    if i >= nb_plan:
        fig.delaxes(ax[y][x])
    else:
        ax[y][x].imshow(waves_5[i], cmap="gray")
        ax[y][x].set_axis_off()
plt.plot()
plt.show()
