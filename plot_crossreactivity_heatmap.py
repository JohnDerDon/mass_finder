import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from math import pi
from matplotlib.ticker import FormatStrFormatter

def plot_heatmap(mean_csv, sum_intensity_csv, std_dev_csv):
    # Load data from CSV files with proper error handling
    try:
        mean_mod_rates_df = pd.read_csv(mean_csv, index_col=0, delimiter=';')
        sum_intensity_df = pd.read_csv(sum_intensity_csv, index_col=0, delimiter=';')
        std_dev_csv_df = pd.read_csv(std_dev_csv, index_col=0, delimiter=';')
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    # Quality control: Print the DataFrames
    print("\nLoaded Mean Modification Rates DataFrame:")
    print(mean_mod_rates_df)
    print("\nLoaded Sum Intensity DataFrame:")
    print(sum_intensity_df)
    print("\nLoaded Standard Deviation DataFrame:")
    print(std_dev_csv_df)

    # Check for NaN values in the dataframes
    if mean_mod_rates_df.isnull().any().any() or sum_intensity_df.isnull().any().any() or std_dev_csv_df.isnull().any().any():
        print("Warning: Found NaN values in the input data.")
        mean_mod_rates_df = mean_mod_rates_df.fillna(0)
        sum_intensity_df = sum_intensity_df.fillna(0)
        std_dev_csv_df = std_dev_csv_df.fillna(0)

    # Ensure same shape of all DataFrames
    if mean_mod_rates_df.shape != sum_intensity_df.shape or mean_mod_rates_df.shape != std_dev_csv_df.shape:
        print("Error: Input CSV files must have the same shape.")
        return

    # Convert DataFrames to numpy arrays
    mean_mod_rates = mean_mod_rates_df.to_numpy()
    sum_intensity = sum_intensity_df.to_numpy()
    std_dev = std_dev_csv_df.to_numpy()

    # Set zero or negative values in sum_intensity to a very small positive value
    sum_intensity_safe = np.where(sum_intensity <= 0, 1e-10, sum_intensity)

    # Compute log10 of sum intensity
    log_sum_intensity = np.log10(sum_intensity_safe)

    # Define colors for peptides and enzymes directly as dictionaries
    color_map = {
        'ane': '#C0C0C0',
        'api': '#C39BE1',
        'ksp': '#BA9B88',
        'mpr': '#F2A068',
        'npu': '#8BC167',
        'pal': '#00CC66',
        'pba': '#C3C02F',
        'pbh': '#E7E200',
        'pha': '#EB6B6B',
        'plc': '#6666FE',
        'xyp': '#53C9D5'
    }

    # Create the main plot and an additional subplot for the legend
    fig, (ax, ax_legend) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]}, figsize=(18, 10))

    # Existing heatmap plotting on ax (left subplot)
    ax.set_xticks(np.arange(mean_mod_rates.shape[1]) + 0.5)
    ax.set_yticks(np.arange(mean_mod_rates.shape[0]) + 0.5)

    ax.tick_params(axis='x', which='major', direction='out', bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(axis='y', which='major', direction='out', left=False, labelleft=True)

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
    x_colors = [
        next((color_map[key] for key in color_map if key.lower() in col.lower()), 'black')
        for col in mean_mod_rates_df.columns
    ]
    ax.set_xticks(np.arange(mean_mod_rates.shape[1]) + 0.5)
    ax.set_xticklabels(mean_mod_rates_df.columns, fontsize=24, weight='bold', rotation=30)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(x_colors[i])

    y_colors = [
        next((color_map[key] for key in color_map if key.lower() in idx.lower()), 'black')
        for idx in mean_mod_rates_df.index
    ]
    ax.set_yticks(np.arange(mean_mod_rates.shape[0]) + 0.5)
    ax.set_yticklabels(mean_mod_rates_df.index, fontsize=24, weight='bold')
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(y_colors[i])

    # Calculate the width of each square (data point) on the heatmap
    # Get the corners of a single cell in display coordinates
    x0, y0 = ax.transData.transform((0, 0))
    x1, y1 = ax.transData.transform((1, 1))  # 1 unit over and 1 unit up

    # Cell width and height in pixels
    cell_width_px = abs(x1 - x0)
    cell_height_px = abs(y1 - y0)

    # Convert pixels to points (1 point = 1/72 inch, fig.dpi pixels per inch)
    cell_width_points = cell_width_px * 72.0 / fig.dpi
    cell_height_points = cell_height_px * 72.0 / fig.dpi

    # Take the smaller one as the maximum diameter
    max_diameter_points = max(cell_width_points, cell_height_points)

    # Scatter `s` is area in points^2, so radius = max_diameter_points / 2
    max_radius_points = max_diameter_points / 2

    # Normalize circle sizes and scale them based on log_sum_intensity
    cmap = plt.colormaps["Greys"]  # Use a grayscale colormap

    # Plot circles based on mean (color) and sum intensity (size)
    for i in range(mean_mod_rates.shape[0]):
        for j in range(mean_mod_rates.shape[1]):
            mean_val = mean_mod_rates[i, j]  # color is based on mean conversion rate
            intensity_val = sum_intensity[i, j]  # circle size is based on sum intensity
            std_dev_val = std_dev[i, j]  # line width based on standard deviation

            # Grayscale color based on mean conversion rate
            color = cmap(mean_val)

            # Handle invalid/zero values safely for log10
            if intensity_val <= 0:
                radius_intensity_factor = radius_std_dev_factor = 0.0
            else:
                log_intensity_val = np.log10(intensity_val)  # log10 scaling
                # Clamp to [0,10] since max log10 is defined as 10 in your spec
                log_intensity_val = min(max(log_intensity_val, 0), 10)
                radius_intensity_factor = log_intensity_val / 10.0  # normalize into [0,1]
                radius_std_dev_factor = (1 - np.sqrt(std_dev_val)) * radius_intensity_factor

            radius_intensity_points = max_radius_points * radius_intensity_factor
            radius_std_dev_points = max_radius_points * radius_std_dev_factor

            size_intensity = pi * (radius_intensity_points ** 2)
            size_std_dev = pi * (radius_std_dev_points ** 2)

            # Draw edge if size > 0
            # edge size is based on sum intensity
            if size_intensity > 0:
                ax.scatter(j + 0.5, i + 0.5, s=size_intensity,
                           color="none", edgecolor="black", linewidth=0.2)

            # Draw circle if size > 0
            # circle size is based on sum intensity and standard deviation
            if size_std_dev > 0:
                ax.scatter(j + 0.5, i + 0.5, s=size_std_dev,
                           color=color, edgecolor='none')

    # Legend for circle standard deviation sizes
    ax_legend.axis([0, 1, 0, 1])
    ax_legend.axis('off')
    ax_legend.text(0.5, 0.96, "Standard\nDeviation", fontsize=16, ha='center')

    example_sum_value = 10**8  # fixed value for std dev legend
    normalized_example_sum_value = np.log10(example_sum_value) / 10.0
    radius_norm_sum = max_radius_points * normalized_example_sum_value
    size_sum = pi * (radius_norm_sum ** 2)
    example_std_dev_values = [0.2, 0.05, 0.01]  # example values to cover log range
    legend_y_positions = [0.90, 0.775, 0.65]

    for i, std_dev_value in enumerate(example_std_dev_values):
        radius_norm_std_dev_points = max_radius_points * (normalized_example_sum_value * (1 - np.sqrt(std_dev_value)))
        size = pi * (radius_norm_std_dev_points ** 2)
        ax_legend.scatter(0.5, legend_y_positions[i], s=size_sum, facecolor='none', edgecolor='black', linewidth=0.2)
        ax_legend.scatter(0.5, legend_y_positions[i], s=size, facecolor='black', edgecolor='none')
        ax_legend.text(0.5, legend_y_positions[i]-0.07, f"{std_dev_value:.2f}", fontsize=16, ha='center')

    # Create color legend for mean values (grayscale)
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.024)
    cbar.ax.set_title('Conversion\nRate', fontsize=16, y=1.025)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Legend for circle intensity sizes
    ax_legend.axis([0, 1, 0, 1])
    ax_legend.axis('off')
    ax_legend.text(0.5, 0.41, "Sum\nIntensity", fontsize=16, ha='center')

    example_sum_values = [10**9, 10**7, 10**5]  # example values to cover log range
    normalized_example = np.log10(example_sum_values) / 10.0
    legend_y_positions = [0.35, 0.225, 0.1]

    for i, (norm_val, sum_val) in enumerate(zip(normalized_example, example_sum_values)):
        radius_norm_points = max_radius_points * norm_val
        size = pi * (radius_norm_points ** 2)
        ax_legend.scatter(0.5, legend_y_positions[i], s=size, facecolor='none', edgecolor='black', linewidth=0.2)
        ax_legend.text(0.5, legend_y_positions[i]-0.07, f"{sum_val:.0e}".replace("+0", ""), fontsize=16, ha='center')


    plt.tight_layout()
    plt.savefig("heatmap.svg", format='svg')
    #plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate a heatmap from two CSV files.")
    parser.add_argument('-mean_csv', type=str, required=True,
                        help="Path to the CSV file containing mean modification rates.")
    parser.add_argument('-sum_intensity_csv', type=str, required=True,
                        help="Path to the CSV file containing sum intensity values.")
    parser.add_argument('-std_dev_csv', type=str, required=True,
                        help="Path to the CSV file containing standard deviation values.")
    args = parser.parse_args()
    plot_heatmap(args.mean_csv, args.sum_intensity_csv, args.std_dev_csv)


if __name__ == '__main__':
    main()
