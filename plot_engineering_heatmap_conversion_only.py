import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from math import pi
from matplotlib.ticker import FormatStrFormatter

def plot_heatmap(mean_csv, data_qual_csv):
    # Load data from CSV files with proper error handling
    try:
        mean_mod_rates_df = pd.read_csv(mean_csv, index_col=0, delimiter=';')
        data_qual_csv_df = pd.read_csv(data_qual_csv, index_col=0, delimiter=';')
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    # Quality control: Print the DataFrames
    print("\nLoaded Mean Modification Rates DataFrame:")
    print(mean_mod_rates_df)
    print("\nLoaded Sum Intensity DataFrame:")
    print(data_qual_csv_df)

    # Check for NaN values in the dataframes
    if mean_mod_rates_df.isnull().any().any() or data_qual_csv_df.isnull().any().any():
        print("Warning: Found NaN values in the input data.")
        mean_mod_rates_df = mean_mod_rates_df.fillna(0)
        data_qual_csv_df = data_qual_csv_df.fillna(0)

    dfs = [mean_mod_rates_df, data_qual_csv_df]
    if not all(df.shape == dfs[0].shape for df in dfs):
        print("Error: Input CSV files must have the same shape.")
        return

    # Convert DataFrames to numpy arrays
    mean_mod_rates = mean_mod_rates_df.to_numpy()
    data_qual = data_qual_csv_df.to_numpy()

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
    fig, ax = plt.subplots(figsize=(14, 10))

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
    ax.set_xticklabels(mean_mod_rates_df.columns, fontsize=24, weight='bold', rotation=45)
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(x_colors[i])

    y_colors = [
        next((color_map[key] for key in color_map if key.lower() in idx.lower()), 'black')
        for idx in mean_mod_rates_df.index
    ]
    ax.set_yticks(np.arange(mean_mod_rates.shape[0]) + 0.5)
    ax.set_yticklabels(mean_mod_rates_df.index, fontsize=24)
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

    # Marker mapping for data quality
    quality_marker_map = {
        1: '',    # No data available
        2: '*',    # Triangle for low quality
    }

    # Plot circles based on mean (color) and sum intensity (size)
    for i in range(mean_mod_rates.shape[0]):
        for j in range(mean_mod_rates.shape[1]):
            mean_val = mean_mod_rates[i, j]  # color is based on mean conversion rate
            qual_val = data_qual[i, j]  # use data quality for indication markers

            # Grayscale color based on mean conversion rate
            color = cmap(mean_val)

            # Handle invalid/zero values safely for log10 or quality values not 0
            if qual_val!= 0:
                radius = 0 * max_radius_points
            else:
                radius = 0.9 * max_radius_points

            size_circle = pi * (radius ** 2)

            # Draw circle if size > 0
            # circle size is based on sum intensity and standard deviation
            if radius > 0:
                ax.scatter(j + 0.5, i + 0.5, s=size_circle,
                           color=color, edgecolor='grey')

            # Draw marker if qual_val != 0. 0 indicates high quality, so no marker.
            if qual_val in quality_marker_map:
                ax.scatter(j + 0.5, i + 0.5, s=100,  # fixed size for visibility
                           color='black', marker=quality_marker_map[qual_val], edgecolor='none')

    # Create color legend for mean values (grayscale)
    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.03, shrink=0.8, aspect=30)
    cbar.ax.set_title('conversion', fontsize=20, y=-2)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    plt.tight_layout()
    plt.savefig("heatmap.svg", format='svg')
    #plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate a heatmap from two CSV files.")
    parser.add_argument('-mean_csv', type=str, required=True,
                        help="Path to the CSV file containing mean modification rates.")
    parser.add_argument('-data_qual_csv', type=str, required=True,
                        help="Path to the CSV file containing data quality indicators.")
    args = parser.parse_args()
    plot_heatmap(args.mean_csv, args.data_qual_csv)


if __name__ == '__main__':
    main()
