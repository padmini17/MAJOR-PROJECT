from PIL import Image,ImageDraw                
from os import path
from termcolor import cprint        
from pyfiglet import figlet_format    
from rich import print                       
from rich.console import Console                                              
import getpass                           
import sys
import os
import binascii                                   
import time             
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import matplotlib.pyplot as plt
from tabulate import tabulate
import subprocess
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from tabulate import tabulate
import cv2
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint
import subprocess


def display_main_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n\n\n")
    
    # Print Major Project heading
    cprint("MAJOR PROJECT".center(os.get_terminal_size().columns), 'red', attrs=['bold'])
    print("\n\n")
    
    # Print Team heading
    cprint("TEAM 1".center(os.get_terminal_size().columns), 'yellow', attrs=['bold'])
    print("\n\n")

    # Print Topic and project details
    
    cprint("PROJECT : ENHANCED IMAGE STEGANOGRAPHY WITH DUAL AUTHENTICATION".center(os.get_terminal_size().columns), 'cyan')
    cprint("AND CAMELLIA CIPHER ENCRYPTION".center(os.get_terminal_size().columns), 'cyan')
    
    print("\n\n\n")
    
    cprint(" STAGE 1: COVER IMAGE SELECTION BASED ON IMAGE SIMILARITY & EMBEDDING DISTORTION ".center(os.get_terminal_size().columns), 'green')
    print("\n\n")
    cprint(" EXPERIMENT 3:  FINAL COVER IMAGE SELECTION BASED ON IMAGE SIMILARITY & EMBEDDING DISTORTION ".center(os.get_terminal_size().columns), 'green')
    print("\n\n")
    # Wait for user input to continue
    input("")



def display_images(images, image_names):
    fig, axs = plt.subplots(1, len(images), figsize=(12, 6))
    fig.suptitle("Two Selected Images", fontsize=20, fontname='fantasy')

    for i, (image, image_name) in enumerate(zip(images, image_names)):
        ax = axs[i]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{image_name}')
        ax.axis('off')

    plt.show()


def calculate_color_histogram(image):
    channels = image.shape[2] if len(image.shape) == 3 else 1
    channel_ranges = [0, 256] * channels
    hist = cv2.calcHist([image], list(range(channels)), None, [256] * channels, channel_ranges)
    hist = hist.flatten()
    return hist



# Function to calculate entropy
def calculate_entropy(image):
    hist = calculate_color_histogram(image)
    return entropy(hist)

# Function to calculate fractal dimension without using a library
def calculate_fractal_dimension(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)[1]
    
    # Count the number of white pixels (foreground) and black pixels (background)
    foreground_pixels = np.sum(threshold_image == 255)
    background_pixels = np.sum(threshold_image == 0)
    
    # Check for division by zero
    if foreground_pixels == 0 or background_pixels == 0:
        return 0.0
    
    # Calculate the fractal dimension
    fractal_dimension = np.log10(foreground_pixels) / np.log10(foreground_pixels + background_pixels)
    return fractal_dimension

# Function to calculate edge detection using Canny
def calculate_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)
    edge_pixels = np.sum(edges == 255)
    total_pixels = image.shape[0] * image.shape[1]
    edge_density = edge_pixels / total_pixels
    return edge_density, edges  # Return both the density and the edge map

def mark_high_feature_areas(selected_images, edge_maps, image_names, threshold=0.1, feature_name='Feature'):
    fig, axs = plt.subplots(2, len(selected_images), figsize=(20, 10))
    fig.suptitle(f"{feature_name} Visualization for Selected Images", fontsize=20, fontname='fantasy')

    for i, (image, edge_map, image_name) in enumerate(zip(selected_images, edge_maps, image_names)):
        row = i // len(selected_images)
        col = i % len(selected_images)

        # Display the original image with marked edges
        ax_original = axs[row, col]
        ax_original.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_original.set_title(f'{image_name}\n')
        ax_original.axis('off')

        # Threshold the edge map to obtain high edge areas
        high_edge_areas = edge_map > threshold

        # Create a copy of the original image to overlay the edges
        marked_image = image.copy()

        # Mark areas with edge density above the threshold in red
        marked_image[high_edge_areas, 0] = 0  # Set the blue channel to 0
        marked_image[high_edge_areas, 1] = 0  # Set the green channel to 0
        marked_image[high_edge_areas, 2] = 255  # Set the red channel to 255

        # Display the marked edges on the original image
        ax_original.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))

        # Additional subplot for edge detection map below the original image
        ax_edge_map = axs[row + 1, col]
        ax_edge_map.set_title(f'Edge Detection Map\n')
        ax_edge_map.imshow(edge_map, cmap='gray')  # Display the edge detection map
        ax_edge_map.axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.5)
    plt.show()

# Function to display the final selected image
def display_final_selected_image(selected_images, selected_image_names, cumulative_metrics):
    highest_cumulative_index = np.argmax(cumulative_metrics)
    highest_cumulative_image = selected_images[highest_cumulative_index]
    highest_cumulative_image_name = selected_image_names[highest_cumulative_index]

    # Display the image with the highest cumulative metric
    print("\n")
    cprint(f'Final Selected Cover Image between 2 Images is : {highest_cumulative_image_name}','green')
    print("\n")
    plt.figure(figsize=(6,6))
    highest_cumulative_image_rgb = cv2.cvtColor(highest_cumulative_image, cv2.COLOR_BGR2RGB)
    # Display image with 'nearest' interpolation for clarity
    plt.imshow(highest_cumulative_image_rgb, interpolation='nearest')

    plt.title(f'Final Selected Cover Image for Steganography: {highest_cumulative_image_name}\n',fontname='fantasy',fontsize=30)
    plt.axis('off')
    plt.show()

def display_entropy_table(image_names, entropy_values):
    cprint('Entropy values for the Selected Images','yellow') 
    table_data = [[''] + image_names, ['Entropy'] + [f'{value:.4f}' for value in entropy_values]]
    table = tabulate(table_data, headers='firstrow', tablefmt='fancy_grid', numalign='center')
    print(table)

# Function to display fractal dimension values in table format
def display_fractal_dimension_table(image_names, fractal_dimensions):
    cprint('Fractal Dimension values for the Selected Images','yellow') 
    table_data = [[''] + image_names, ['Fractal Dimension'] + [f'{value:.4f}' for value in fractal_dimensions]]
    table = tabulate(table_data, headers='firstrow', tablefmt='fancy_grid', numalign='center')
    print(table)

# Function to display edge detection values in table format
def display_edge_detection_table(image_names, edge_detection_values):
    cprint('Edge Detection values for the Selected Images','yellow') 
    table_data = [[''] + image_names, ['Edge Detection'] + [f'{value[0]:.4f}' for value in edge_detection_values]]
    table = tabulate(table_data, headers='firstrow', tablefmt='fancy_grid', numalign='center')
    print(table)

def calculate_cumulative_metrics(selected_images, image_names):
    entropy_values = [calculate_entropy(image) for image in selected_images]
    fractal_dimensions = [calculate_fractal_dimension(image) for image in selected_images]
    edge_detection_values = [calculate_edge_detection(image) for image in selected_images]

    cumulative_metrics = [entropy + edge_detection[0] + fractal_dimension for entropy, edge_detection, fractal_dimension in zip(entropy_values, edge_detection_values, fractal_dimensions)]

    return entropy_values, fractal_dimensions, edge_detection_values, cumulative_metrics


def display_cumulative_metrics_table(image_names, entropy_values, fractal_dimensions, edge_detection_values, cumulative_metrics):
    cprint('\nCumulative Metrics for the Selected Images\n','yellow')
    table_data = [[''] + image_names, 
                  ['Entropy'] + [f'{value:.4f}' for value in entropy_values],
                  ['Fractal Dimension'] + [f'{value:.4f}' for value in fractal_dimensions],
                  ['Edge Detection'] + [f'{value[0]:.4f}' for value in edge_detection_values],
                  ['Cumulative Metrics'] + [f'{value:.4f}' for value in cumulative_metrics]]
    table = tabulate(table_data, headers='firstrow', tablefmt='fancy_grid', numalign='center')
    print(table)



# Function to execute the main process
# Function to execute the main process
def main():
    display_main_screen()
    # Example usage:
    image_paths = ["image2.png", "image11_compressed.png"]

    images = [cv2.imread(path) for path in image_paths]
    image_names = [f'Image {i + 1}' for i in range(len(images))]

    # Resize images to the same dimensions
    min_height = min(image.shape[0] for image in images)
    min_width = min(image.shape[1] for image in images)
    selected_images = [cv2.resize(image, (min_width, min_height)) for image in images]

    # Display selected images
    display_images(selected_images, image_names)

    # Calculate and display metrics
    entropy_values, fractal_dimensions, edge_detection_values, cumulative_metrics = calculate_cumulative_metrics(selected_images, image_names)

# Display cumulative metrics table

    display_entropy_table(image_names, entropy_values)
    display_fractal_dimension_table(image_names, fractal_dimensions)
    display_edge_detection_table(image_names, edge_detection_values)

    # Display high feature areas
    mark_high_feature_areas(selected_images, [edge_map for _, edge_map in edge_detection_values], image_names, threshold=0.1, feature_name='Edge Detection')

    # Display cumulative metrics table
    display_cumulative_metrics_table(image_names, entropy_values, fractal_dimensions, edge_detection_values, cumulative_metrics)
    # Display the final selected image
    display_final_selected_image(selected_images, image_names, cumulative_metrics)
    input("")
    subprocess.run(["python","project2.py"])

if __name__ == "__main__":
    main()


    
