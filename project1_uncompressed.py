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
    cprint(" EXPERIMENT 2: COVER IMAGE SELECTION AMONG UNCOMPRESSED PNG IMAGES ".center(os.get_terminal_size().columns), 'green')
    print("\n\n")
    # Wait for user input to continue
    input("")

# Call the function to display the main screen
display_main_screen()



def main():
    display_main_screen()

# Function to calculate color histogram
def calculate_color_histogram(image):
    channels = image.shape[2] if len(image.shape) == 3 else 1
    channel_ranges = [0, 256] * channels
    hist = cv2.calcHist([image], list(range(channels)), None, [256] * channels, channel_ranges)
    hist = hist.flatten()
    return hist

def display_all_images(images, image_names,original_image_paths):
    fig, axs = plt.subplots(2, 4, figsize=(16, 12))
    fig.suptitle("Cover Image Selection from these Batch Images", fontsize=30,fontname='fantasy')

    for i, (image, image_name,original_path) in enumerate(zip(images, image_names,original_image_paths)):
        ax = axs[i // 4, i % 4]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{os.path.basename(image_name)}')
        ax.axis('off')

    # Manually adjust spacing
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.show()


def display_selected_images(images, image_names):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("4 Selected Images Based on Least Cumulative SSIM Value", fontsize=20, fontname='fantasy')

    for i, (image, image_name) in enumerate(zip(images, image_names)):
        ax = axs[i]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(image_name)
        ax.axis('off')

    # Manually adjust spacing
    plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.2, wspace=0.3, hspace=0.3)
    plt.show()

def display_individual_histograms_and_statistics(images, titles):
    for i, image in enumerate(images):
        plt.figure(figsize=(16, 8))

        # Calculate separate histograms for each channel
        channel_histograms = [calculate_color_histogram(image[:, :, j]) for j in range(image.shape[2])]

        # Display the image in the center
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')  # Hide axes for better presentation

        # Display additional statistics in a table beside the image
        plt.subplot(2, 3, 2)
        red_values = image[:, :, 0].ravel()
        green_values = image[:, :, 1].ravel()
        blue_values = image[:, :, 2].ravel()

        total_red_channels = np.sum(red_values)
        total_green_channels = np.sum(green_values)
        total_blue_channels = np.sum(blue_values)

        additional_statistics = [
            ['Statistic Name', 'Value'],
            ['Height of the image', image.shape[0]],
            ['Width of the image', image.shape[1]],
            ['Number of Pixels in the image', image.size],
            ['Number of Channels in the image', image.shape[2]],
            ['Intensity of Red Channels', total_red_channels],
            ['Intensity of Green Channels', total_green_channels],
            ['Intensity of Blue Channels', total_blue_channels],
            ['Most Occurring Color in the image', ['Red', 'Green', 'Blue'][np.argmax([np.sum(red_values), np.sum(green_values), np.sum(blue_values)])]],
            ['Mean Red Value in the Image', f'{np.mean(red_values):.3f}'],
            ['Mean Green Value in the Image', f'{np.mean(green_values):.3f}'],
            ['Mean Blue Value in the Image', f'{np.mean(blue_values):.3f}'],
        ]

        # Flatten the nested lists
        flattened_statistics = [item for sublist in additional_statistics for item in sublist]

        # Convert the flattened list to a 1D array
        flattened_array = np.array(flattened_statistics).reshape(-1, 2)

        # Create the table with adjusted column width
        plt.rcParams['font.family'] = 'serif'
        
        table = plt.table(cellText=flattened_array, colLabels=None, loc='center',
                          cellLoc='center', colColours=['red']*2,
                          cellColours=[['white', 'white']] * len(flattened_array), colWidths=[0.8, 0.3],
                          fontsize=18)  # Adjust the column width and font size

        # Set the title of the table
        title = plt.title('Color Histograms & Statistics of the Image', fontsize=30,y=1,pad=30,fontname='fantasy')

        # Hide axes for better presentation
        plt.axis('off')

        # Display Red histogram
        plt.subplot(2, 3, 4)
        plt.plot(channel_histograms[0], color='red')
        plt.title('Red Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        # Display Green histogram
        plt.subplot(2, 3, 5)
        plt.plot(channel_histograms[1], color='green')
        plt.title('Green Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        # Display Blue histogram
        plt.subplot(2, 3, 6)
        plt.plot(channel_histograms[2], color='blue')
        plt.title('Blue Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplots_adjust(top=0.85)

        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()



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

def mark_high_feature_areas(images, edge_maps, image_names, threshold=0.1, feature_name='Feature'):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{feature_name} Visualization for Selected Images", fontsize=20, fontname='fantasy')

    for i, (image, edge_map, image_name) in enumerate(zip(images, edge_maps, image_names)):
        row = i // 4
        col = i % 4

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


# Function to display color histograms
def display_color_histograms(images, titles):
    plt.figure(figsize=(16, 8))

    for i, image in enumerate(images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(titles[i])

        # Calculate separate histograms for each channel
        channel_histograms = [calculate_color_histogram(image[:, :, j]) for j in range(image.shape[2])]
        for color, hist in zip(['Red', 'Green', 'Blue'], channel_histograms):
            plt.subplot(2, 5, i + 6)
            plt.plot(hist, color=color[0].lower())
            plt.title(f'{color} Histogram')

    plt.show()

# Function to display SSIM values in table format
def display_ssim_table(image_names, ssim_values):
    table_data = [[''] + image_names] + [[image_names[i]] + [f'{value:.4f}' for value in ssim_values[i]] for i in range(len(image_names))]
    table = tabulate(table_data, headers='firstrow', tablefmt='fancy_grid', numalign='center')
    print(table)

# Function to display entropy values in table format
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

def calculate_cumulative_metrics(selected_images, selected_image_names):
    entropy_values = [calculate_entropy(image) for image in selected_images]
    fractal_dimensions = [calculate_fractal_dimension(image) for image in selected_images]
    edge_detection_values = [calculate_edge_detection(image) for image in selected_images]

    cumulative_metrics = [entropy + edge_detection[0] + fractal_dimension for entropy, edge_detection, fractal_dimension in zip(entropy_values, edge_detection_values, fractal_dimensions)]

    return cumulative_metrics

def display_cumulative_metrics_table(image_names, entropy_values, fractal_dimensions, edge_detection_values, cumulative_metrics):
    cprint('\nCumulative Metrics for the Selected Images\n','yellow')
    table_data = [[''] + image_names, 
                  ['Entropy'] + [f'{value:.4f}' for value in entropy_values],
                  ['Fractal Dimension'] + [f'{value:.4f}' for value in fractal_dimensions],
                  ['Edge Detection'] + [f'{value[0]:.4f}' for value in edge_detection_values],
                  ['Cumulative Metrics'] + [f'{value:.4f}' for value in cumulative_metrics]]
    table = tabulate(table_data, headers='firstrow', tablefmt='fancy_grid', numalign='center')
    print(table)


def main():
    display_main_screen()

    # Example usage:
    image_paths = ["image1.png", "image2.png", "image3.png", "image4.png", "image5.png", "image6.png", "image7.png", "image8.png"]

    images = [cv2.imread(path) for path in image_paths]
    image_names = [f'Image {i + 1}' for i in range(len(images))]

    # Resize images to the same dimensions
    min_height = min(image.shape[0] for image in images)
    min_width = min(image.shape[1] for image in images)
    images_resized = [cv2.resize(image, (min_width, min_height)) for image in images]

    cprint("COLOR HISTOGRAMS & STATISTICS TABLE: \n",'red')
    print(" Displaying the 8 Batch Images along with their Color Histograms & Statistics table")
    print("\n")


    # Display all images
    display_all_images(images_resized, image_names,image_paths)

    # Display color histograms
    display_individual_histograms_and_statistics(images_resized[:8], image_names[:8])
    ssim_values = [[ssim(images_resized[i], images_resized[j], full=True, win_size=3)[0] for j in range(len(images_resized))] for i in range(len(images_resized))]
    print("\n\n")
    # Display SSIM values
    cprint("IMAGE SIMILARITY: \n",'red')
    print("\n Image Similarity by SSIM\n")
    print(" Calculating Structural Similarity Index (SSIM) for all the 8 Images ")
    print("\n")
    cprint("Structural Similarity Index (SSIM) Values for the Images",'yellow')
    display_ssim_table(image_names, ssim_values)

    # Calculate cumulative SSIM values for each image
    print("\n")
    print(" Calculating the Cumulative SSIM Value for Each Image ")
    cumulative_ssim_values = [np.sum(ssim_values[i]) - ssim_values[i][i] for i in range(len(images_resized))]

    # Display cumulative SSIM values in a table
    cumulative_ssim_table = [[''] + image_names, ['Cumulative SSIM'] + [f'{value:.4f}' for value in cumulative_ssim_values]]
    table = tabulate(cumulative_ssim_table, headers='firstrow', tablefmt='fancy_grid', numalign='center')
    print(table)

    selected_indices = np.argsort(cumulative_ssim_values)[:4]
    selected_images = [images_resized[i] for i in selected_indices]
    selected_image_names = [image_names[i] for i in selected_indices]

    # Find the indices of the 4 images with the lowest SSIM values
    lowest_ssim_indices = np.argsort(cumulative_ssim_values)[:4]
    lowest_ssim_images = [images_resized[i] for i in lowest_ssim_indices]
    lowest_ssim_image_names = [image_names[i] for i in lowest_ssim_indices]

    # Prepare data for the table of 4 images with the lowest SSIM values
    lowest_ssim_table_data = [['Image Name', 'SSIM Value']]
    lowest_ssim_values = [cumulative_ssim_values[i] for i in lowest_ssim_indices]
    for i in range(4):
        image_name = lowest_ssim_image_names[i]
        ssim_value = lowest_ssim_values[i]
        lowest_ssim_table_data.append([image_name, f'{ssim_value:.4f}'])

    print("\n\n At this stage, we select only 4 Images based on their least Cumulative SSIM Value")
    # Display the table of 4 images with the lowest SSIM values
    print(tabulate(lowest_ssim_table_data, headers='firstrow', tablefmt='fancy_grid', numalign='center'))

    display_selected_images(lowest_ssim_images, lowest_ssim_image_names)

    cprint("\nEMBEDDING DISTORTION: \n",'red')
    print(" Calculating Embedding Distortion based on Entropy, Fractal Dimension and Edge Detection \n")

    #Prepare data for the table
    table_data = [['Image Name', 'Entropy', 'Fractal Dimension', 'Edge Detection', 'Cumulative Metrics']]

    # Calculate entropy, fractal dimension, and edge detection for the selected images
    selected_entropy_values = [calculate_entropy(image) for image in selected_images]
    selected_fractal_dimensions = [calculate_fractal_dimension(image) for image in selected_images]
    selected_edge_detection_values = [calculate_edge_detection(image) for image in selected_images]

    # Display the entropy values in a table

    display_entropy_table(selected_image_names, selected_entropy_values)
    print("\n")

    # Display the fractal dimension values in a table

    display_fractal_dimension_table(selected_image_names, selected_fractal_dimensions)
    print("\n")

    # Display the edge detection values in a table

    display_edge_detection_table(selected_image_names, selected_edge_detection_values)
    print("\n")

    # Change this line in the main function
    mark_high_feature_areas(selected_images, [edge_map for _, edge_map in selected_edge_detection_values], selected_image_names, threshold=0.1, feature_name='Edge Detection')


    cumulative_metrics = calculate_cumulative_metrics(selected_images, selected_image_names)

    # Display cumulative metrics values in a table
    print("\nDisplaying the cumulative metrics for the embedding distortion")
    display_cumulative_metrics_table(selected_image_names, selected_entropy_values, selected_fractal_dimensions, selected_edge_detection_values, cumulative_metrics)
    print("\n")
    # Find the index of the image with the highest cumulative metric
    highest_cumulative_index = np.argmax(cumulative_metrics)
    highest_cumulative_image = selected_images[highest_cumulative_index]
    highest_cumulative_image_name = selected_image_names[highest_cumulative_index]


    # Display the image with the highest cumulative metric
    print("\n")
    cprint(f'Selected Cover Image from 8 Uncompressed PNG Images is : {highest_cumulative_image_name}','green')
    print("\n")
    plt.figure(figsize=(6,6))
    highest_cumulative_image_rgb = cv2.cvtColor(highest_cumulative_image, cv2.COLOR_BGR2RGB)
    # Display image with 'nearest' interpolation for clarity
    plt.imshow(highest_cumulative_image_rgb, interpolation='nearest')

    plt.title(f'Selected Cover Image : {highest_cumulative_image_name}\n',fontname='fantasy',fontsize=30)

    plt.axis('off')
    plt.show()
    cprint("\n Now, executing the Next Process: Final Selection of the Cover Image",'red')
    print("\n")
    input("")
    subprocess.run(["python", "project1_final.py"])


if __name__== "__main__":
    main()


    
