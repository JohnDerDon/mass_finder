# -*- coding: utf-8 -*-
"""
Parse through mzXML files to plot the intensities of a mass spectrum of a specific mass range in a given time frame.

(C) Johannes Eckert, Mathijs Mabesoone, ETH Zurich
April 2024
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyteomics import mzxml
import multiprocess as mp


def append_suffix_to_file(file, overwrite):
    """
    Check if the specified file exists, and if so, append a suffix to the filename.
    :param file: Base file name.
    :param overwrite: Boolean indicating whether to overwrite existing files.
    :return: Updated file name.
    """
    file = os.path.splitext(file)[0]

    if overwrite:
        return file

    suffix = 1
    while any([os.path.isfile(f"{file}_{suffix}{ext}") for ext in ['_mass_spectrum.svg']]):
        suffix += 1
    return f"{file}_{suffix}"


def analyze_mass_spec(spectrum, mass_range, time_range, min_intensity):
    """
    Analyze a single mass spectrum to filter based on mass range, retention time, and minimum intensity.
    :param spectrum: The spectrum to analyze.
    :param mass_range: Range of masses to filter.
    :param time_range: Time range to filter.
    :param min_intensity: Minimum intensity threshold.
    :return: A DataFrame of analyzed spectra.
    """
    if spectrum.get('msLevel', 0) != 1:  # Only process MS1 spectra
        return None

    mz_array = np.array(spectrum.get('m/z array', []))
    intensity_array = np.array(spectrum.get('intensity array', []))
    retention_time = spectrum.get('retentionTime', 0)

    if mz_array.size == 0 or intensity_array.size == 0:
        return None

    if not (time_range[0] <= retention_time <= time_range[1]):
        return pd.DataFrame()  # Return early if retention time is out of range

    valid_mask = (mz_array >= mass_range[0]) & (mz_array <= mass_range[1]) & (intensity_array >= min_intensity)

    filtered_masses = mz_array[valid_mask]
    filtered_intensities = intensity_array[valid_mask]

    if len(filtered_masses) == 0:
        return None

    return pd.DataFrame({
        'time': retention_time,
        'experimental_mass': filtered_masses,
        'intensity': filtered_intensities
    })


def generate_plot_dataframe(analyzed_spectra, ppm_range):
    """
    Create a DataFrame for plotting by summing the intensities of masses within the specified ppm range.
    Normalizes the total_intensity values by setting the highest total_intensity to 1.
    Adds a column that indicates whether a mass is the highest intensity in its 0.6 Da range, labeled as True or False.
    :param analyzed_spectra: List of DataFrames containing analyzed spectra.
    :param ppm_range: PPM range for mass binning.
    :return: DataFrame of summed masses, normalized intensities, and a label.
    """
    # Concatenate all spectra and sort by intensity
    all_spectra_dataframe = pd.concat(analyzed_spectra, ignore_index=True)
    all_spectra_dataframe = all_spectra_dataframe.sort_values(by='intensity', ascending=False, ignore_index=True)

    result_rows = []
    used_masses = np.zeros(all_spectra_dataframe.shape[0], dtype=bool)

    # Iterate through each row and calculate total intensity within the specified ppm range
    for index, row in all_spectra_dataframe.iterrows():
        if used_masses[index]:
            continue

        current_mass = row['experimental_mass']
        current_intensity = row['intensity']

        mass_range_low = current_mass * (1 - ppm_range / 1_000_000)
        mass_range_high = current_mass * (1 + ppm_range / 1_000_000)

        # Create mask to select masses within the ppm range
        mask = (all_spectra_dataframe['experimental_mass'] >= mass_range_low) & \
               (all_spectra_dataframe['experimental_mass'] <= mass_range_high)

        # Sum the total intensities and get the highest mass within the range
        total_intensity = all_spectra_dataframe.loc[mask, 'intensity'].sum()
        highest_mass = all_spectra_dataframe.loc[mask, 'experimental_mass'].max()

        result_rows.append({
            'mass': highest_mass,
            'total_intensity': total_intensity
        })

        used_masses[mask] = True

    # Convert result to DataFrame
    result_df = pd.DataFrame(result_rows)

    # Normalize the total_intensity values by dividing by the maximum total_intensity
    max_intensity = result_df['total_intensity'].max()
    result_df['normalized_intensity'] = result_df['total_intensity'] / max_intensity

    # Sort by normalized intensity to determine the highest intensity masses
    result_df = result_df.sort_values(by='normalized_intensity', ascending=False, ignore_index=True)

    # Add a column to indicate if the mass is the highest in its 0.6 Da range
    labels = []
    for i, row in result_df.iterrows():
        current_mass = row['mass']
        if i == 0:
            # The highest intensity mass is always True
            labels.append(True)
            continue

        # Check if there is a higher intensity mass within 0.6 Da range
        within_range = result_df[(result_df['mass'] >= current_mass - 0.6) &
                                 (result_df['mass'] <= current_mass + 0.6) &
                                 (result_df['normalized_intensity'] > row['normalized_intensity'])]

        if within_range.empty:
            labels.append(True)
        else:
            labels.append(False)

    # Add the new column to the DataFrame
    result_df['label'] = labels

    print(result_df)

    return result_df


def plot_results(plot_dataframe, output_file, plot_mass_range, plot_intensity_range, overwrite):
    """
    Plot the analyzed spectra based on the provided DataFrame.
    :param plot_dataframe: DataFrame containing the mass and intensity data to plot.
    :param output_file: Path for saving the plot.
    :param plot_intensity_range: Intensity range for the plot.
    :param plot_mass_range: Mass range for the plot.
    :param overwrite: Boolean to determine if existing plots should be overwritten.
    """
    plt.figure(figsize=(10, 6))

    plt.bar(plot_dataframe['mass'], plot_dataframe['total_intensity'], width=0.5, color='blue', alpha=0.7)
    plt.title('Mass Spectrum')
    plt.xlabel('Mass (m/z)')
    plt.ylabel('Intensity')
    plt.xlim(plot_mass_range)
    plt.ylim(0, plot_dataframe['total_intensity'].max() * 1.1)

    # Customizing ticks
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Add grid and legend
    plt.grid(True)
    plt.legend(['Intensity'])

    # Save plot
    plot_file_path = append_suffix_to_file(output_file, overwrite)
    plt.savefig(f"{plot_file_path}_mass_spectrum.svg")
    plt.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Check mzXML files for specific isotope patterns.')
    parser.add_argument('-input', help='Input folder or file')
    parser.add_argument('-threads', help='Number of threads to use', type=int, default=1)
    parser.add_argument('-min_intensity', help='Minimal intensity of main peak to report. Default = 1e4.', type=float,
                        default=1e4)
    parser.add_argument('-time_range', help='Set a custom time range. Example: 3-10 minutes.', type=str,
                        default='0-1000')
    parser.add_argument('-mass_range', help='Set a custom mass range. Example: 200-600.', type=str, default='200-2000')
    parser.add_argument('-ppm_range', help='Set a custom ppm range for binning masses. Example: 10.', type=str,
                        default='10')
    parser.add_argument('-overwrite', help='If true, existing plots will be overwritten.', action='store_true')
    parser.add_argument('-full_range', help='If true, plot will span the entire time and mass range.',
                        action='store_true')
    parser.add_argument('-plot_intensity_range', help='Intensity range to use for plotting', default='0-1e9', type=str)
    parser.add_argument('-plot_mass_range', help='Mass range to use for plotting', default='200-2000', type=str)
    parser.add_argument('-plot_label_intensity', help='Minimum intensity of the mass labels that is still plotted in %', default='10', type=str)
    parser.add_argument('-plot_group_identifiers', help='Group similar identifiers in the plot with similar colors',
                        default=1, type=int)

    args = parser.parse_args()

    sys.stdout.write(f"{''.join(['=' for _ in range(20)])}\n")

    # Check if the input is a directory or file and make a file list
    if os.path.isdir(args.input):
        mzxml_files = [os.path.join(args.input, file) for file in os.listdir(args.input) if '.mzxml' in file.lower()]
        sys.stdout.write(f"Detected {len(mzxml_files)} .mzXML files in {args.input}:\n")
        sys.stdout.write('\t' + '\n\t'.join(mzxml_files) + '\n')
    elif os.path.isfile(args.input) and '.mzxml' in args.input.lower():
        mzxml_files = [args.input]
        sys.stdout.write(f"Analyzing single file: {args.input}.\n")
    else:
        sys.stdout.write(f"Did not detect any mzXML files in {args.input}. Terminating...\n")
        return

    mass_range = [float(mass) for mass in args.mass_range.split('-')]
    time_range = [float(time) for time in args.time_range.split('-')]
    ppm_range = float(args.ppm_range)

    # Check for full_range logic
    full_range = args.full_range or (args.plot_intensity_range != '0-1e9' or args.plot_mass_range != '200-2000')

    # Analyze each file and write output
    with mp.Pool(args.threads) as pool:
        for input_file in mzxml_files:
            sys.stdout.write(f"Started parsing {input_file}\n")
            data = mzxml.MzXML(input_file, use_index=True)

            min_index, max_index = [int(data.time[float(time)]['id']) for time in time_range]
            analyzed_spectra = pool.starmap(analyze_mass_spec, [
                (data.get_by_index(int(index) - 1), mass_range, time_range, args.min_intensity)
                for index in range(min_index, max_index)
            ])
            analyzed_spectra = [spectrum for spectrum in analyzed_spectra if spectrum is not None]

            sys.stdout.write(
                f"\tFound {sum([len(spectrum) for spectrum in analyzed_spectra])} masses in {input_file}.\n")

            plot_intensity_range = [float(intensity) for intensity in args.plot_intensity_range.split('-')]
            plot_mass_range = [float(mass) for mass in args.plot_mass_range.split('-')]
            plot_dataframe = generate_plot_dataframe(analyzed_spectra, ppm_range)

            sys.stdout.write(f"\tFound {len(plot_dataframe)} unique masses for plotting.\n")
            plot_results(plot_dataframe, os.path.splitext(input_file)[0], plot_mass_range, plot_intensity_range,
                         args.overwrite)

    sys.stdout.write(f"{''.join(['=' for _ in range(20)])}\n")


if __name__ == '__main__':
    main()
