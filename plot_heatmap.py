import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib.cm as cm
from math import pi


def plot_heatmap(mean_csv, std_csv):
    # Load data from CSV files with proper error handling
    try:
        mean_mod_rates_df = pd.read_csv(mean_csv, index_col=0, delimiter=';')
        std_dev_df = pd.read_csv(std_csv, index_col=0, delimiter=';')
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    # Quality control: Print the DataFrames
    print("\nLoaded Mean Modification Rates DataFrame:")
    print(mean_mod_rates_df)
    print("\nLoaded Standard Deviation DataFrame:")
    print(std_dev_df)

    # Check for NaN values in the dataframes
    if mean_mod_rates_df.isnull().any().any() or std_dev_df.isnull().any().any():
        print("Warning: Found NaN values in the input data.")
        mean_mod_rates_df = mean_mod_rates_df.fillna(0)
        std_dev_df = std_dev_df.fillna(0)

    # Ensure same shape
    if mean_mod_rates_df.shape != std_dev_df.shape:
        print("Error: The shape of mean_mod_rates.csv and std_dev.csv do not match.")
        return

    # Convert DataFrames to numpy arrays
    mean_mod_rates = mean_mod_rates_df.to_numpy()
    std_dev = std_dev_df.to_numpy()

    # Set zero values in std_dev to half the minimum non-zero value
    min_std_dev = np.min(std_dev[np.nonzero(std_dev)])  # Get the minimum value excluding zeros
    std_dev = np.where(std_dev == 0, min_std_dev / 2, std_dev)  # Replace 0 values with a small value

    # Convert std_dev to logarithmic values (base 10)
    log_std_dev = np.log10(std_dev)
    min_log_std_dev = np.min(log_std_dev)  # Get the maximum log value for normalization

    # Normalize log_std_dev so that the highest value equals 1
    log_std_dev = log_std_dev / min_log_std_dev


    # Define colors for peptides and enzymes directly as dictionaries
    color_map = {'A': 'blue', 'B': 'green', 'C': 'red', 'D': 'yellow', 'x': 'orange', 'y': 'purple', 'z': 'brown'}

    # Create the plot and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(np.arange(mean_mod_rates.shape[1]) + 0.5)
    ax.set_yticks(np.arange(mean_mod_rates.shape[0]) + 0.5)

    ax.tick_params(axis='x', direction='out', bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(axis='y', direction='out', left=False, labelleft=True)

    ax.set_xlim(0, mean_mod_rates.shape[1])
    ax.set_ylim(0, mean_mod_rates.shape[0])
    ax.invert_yaxis()

    # Create light grey minor gridlines between the major gridlines
    ax.set_xticks(np.arange(mean_mod_rates.shape[1] + 1), minor=True)
    ax.set_yticks(np.arange(mean_mod_rates.shape[0] + 1), minor=True)
    ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_aspect('equal', adjustable='box')

    # Apply dynamic colors to x and y tick labels
    # Generate a list of colors for x-axis labels based on partial matching with column names
    x_colors = [
        next((color_map[key] for key in color_map if key in col), 'black')
        for col in mean_mod_rates_df.columns
    ]
    ax.set_xticks(np.arange(mean_mod_rates.shape[1]) + 0.5)  # Set xticks for the labels
    ax.set_xticklabels(mean_mod_rates_df.columns, fontsize=24, weight='bold', rotation=45, ha='right')

    # Apply colors to x-axis labels directly
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(x_colors[i])

    # Generate a list of colors for y-axis labels based on partial matching with column names
    y_colors = [
        next((color_map[key] for key in color_map if key in idx), 'black')
        for idx in mean_mod_rates_df.index
    ]
    ax.set_yticks(np.arange(mean_mod_rates.shape[0]) + 0.5)  # Set yticks for the labels
    ax.set_yticklabels(mean_mod_rates_df.index, fontsize=24, weight='bold')

    # Apply colors to y-axis labels directly
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(y_colors[i])

    # Calculate the width of each square (data point) on the heatmap
    plot_width = fig.get_size_inches()[0] * fig.dpi  # Get plot width in pixels
    plot_height = fig.get_size_inches()[1] * fig.dpi  # Get plot height in pixels
    square_width = plot_width / mean_mod_rates.shape[1]  # width of each data point square
    square_height = plot_height / mean_mod_rates.shape[0]  # height of each data point square

    # Calculate the maximum allowed diameter (0.5 * square width)
    max_diameter = 0.5 * square_width  # Maximum diameter as 50% of the square width
    max_radius = max_diameter / 2  # Radius corresponding to the max diameter

    # Normalize circle sizes and scale them based on standard deviation
    cmap = cm.get_cmap("Greys")  # Use a grayscale colormap

    # Plot circles based on mean and standard deviation
    for i in range(mean_mod_rates.shape[0]):
        for j in range(mean_mod_rates.shape[1]):
            mean_val = mean_mod_rates[i, j]
            log_sd_val = log_std_dev[i, j]  # Invert the log value for better visualization
            # Grayscale color based on mean value
            color = cmap(mean_val)
            # Calculate the circle size (area) based on standard deviation
            # Formula for area of circle: area = pi * r^2 = pi * (diameter / 2)^2
            size = pi * (max_radius ** 2) * log_sd_val  # Adjust circle size based on standard deviation
            # Plot the circle on the heatmap
            ax.scatter(j + 0.5, i + 0.5, s=size, color=color)

    # Create color legend for mean values (grayscale)
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=0, vmax=1))  # No need to normalize
    sm.set_array([])  # Empty array needed for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax, fraction=0.012, pad=0.04)  # Fraction to control size of colorbar
    cbar.set_label('Conversion Rate')

    plt.tight_layout()
    #plt.savefig("heatmap.svg", format='svg')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate a heatmap from two CSV files.")
    parser.add_argument('-mean_csv', type=str, required=True,
                        help="Path to the CSV file containing mean modification rates.")
    parser.add_argument('-std_dev_csv', type=str, required=True,
                        help="Path to the CSV file containing standard deviations.")
    args = parser.parse_args()
    plot_heatmap(args.mean_csv, args.std_dev_csv)


if __name__ == '__main__':
    main()
