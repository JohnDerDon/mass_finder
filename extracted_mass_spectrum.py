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
import matplotlib.patches as mpatches
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


def generate_plot_dataframe(analyzed_spectra, ppm_binning):
    """
    Create a DataFrame for plotting by summing the intensities of masses within the specified ppm range.
    Normalizes the total_intensity values by setting the highest total_intensity to 1.
    Adds a column that indicates whether a mass is the highest intensity in its 0.6 Da range, labeled as True or False.
    :param analyzed_spectra: List of DataFrames containing analyzed spectra.
    :param ppm_binning: PPM range for mass binning.
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

        mass_range_low = current_mass * (1 - ppm_binning / 1_000_000)
        mass_range_high = current_mass * (1 + ppm_binning / 1_000_000)

        # Create mask to select masses within the ppm range
        mask = (all_spectra_dataframe['experimental_mass'] >= mass_range_low) & \
               (all_spectra_dataframe['experimental_mass'] <= mass_range_high)

        # Sum the total intensities and get the highest mass within the range
        total_intensity = all_spectra_dataframe.loc[mask, 'intensity'].sum()
        highest_mass = all_spectra_dataframe.loc[mask, 'experimental_mass'].max()

        result_rows.append({
            'mass': highest_mass,
            'total_intensity': total_intensity,
            'charge_state': 'x'
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

    print("Result DataFrame:")
    print(result_df)

    return result_df


def generate_plot_dataframe_with_charge_states(plot_dataframe, charge_range, ppm_charge_state, plot_label_intensity):
    """
    Generate a DataFrame with charge states assigned based on the provided DataFrame.
    :param plot_dataframe: DataFrame containing the mass, intensity, and normalized intensity data to plot.
    :param charge_range: Range of charge states to consider.
    :param ppm_charge_state: PPM range for charge state determination.
    :param plot_label_intensity: Minimum intensity threshold for labeling.
    :return: DataFrame with charge states assigned.
    """
    plot_dataframe_charge_states = plot_dataframe.copy()
    # Iterate over peaks to determine charge states
    for index, row in plot_dataframe_charge_states.iterrows():
        if row['charge_state'] != 'x':
            continue  # Skip if this peak has already been assigned a charge state

        if row['normalized_intensity'] > plot_label_intensity and row['label']:
            mz_main = row['mass']
        else:
            continue # Skip if the intensity is below the threshold


        # Check for possible charge states starting from the highest
        for charge in range(charge_range[1], charge_range[0] - 1, -1):
            candidate_peaks = []  # Start with an empty list for candidate peaks
            is_matched = False

            # Append the main peak (highest intensity being evaluated) first
            candidate_peaks.append(row)

            # Look left and right for peaks to match the expected pattern
            for direction in [-1, 1]:  # -1 to look left, 1 to look right
                mz_current = mz_main

                while True:
                    expected_diff = 1.0 / charge
                    mz_expected = mz_current + direction * expected_diff

                    # Check for peaks that are close to the expected m/z value and are unassigned
                    matched_row = plot_dataframe_charge_states[
                        (np.abs(plot_dataframe_charge_states['mass'] - mz_expected) < mz_expected * ppm_charge_state / 1_000_000)
                        & (plot_dataframe_charge_states['charge_state'] == 'x')]

                    if not matched_row.empty:
                        closest_match = matched_row.iloc[0]
                        candidate_peaks.append(closest_match)  # Add to candidate peaks
                        mz_current = closest_match['mass']  # Update to continue searching
                    else:
                        break  # Stop if no match found in this direction

            # Sort candidate peaks by m/z to ensure proper order
            candidate_peaks = sorted(candidate_peaks, key=lambda x: x['mass'])

            print(f"Charge: {charge}, Candidate Peaks:")
            print(candidate_peaks)

            # Ensure no peaks with an existing charge state are in candidate_peaks
            candidate_peaks = [peak for peak in candidate_peaks if plot_dataframe_charge_states.at[peak.name, 'charge_state'] == 'x']

            # Find the peak with the highest intensity in the candidate list
            if not candidate_peaks:
                continue  # No valid candidates, move to the next charge

            highest_intensity_peak = max(candidate_peaks, key=lambda x: x['total_intensity'])
            highest_index = next(i for i, peak in enumerate(candidate_peaks) if peak['mass'] == highest_intensity_peak['mass'])

            # Check if the candidate peaks show a general decreasing intensity trend from the highest point
            if len(candidate_peaks) > 1:
                intensities_left = [peak['total_intensity'] for peak in candidate_peaks[:highest_index + 1][::-1]]
                intensities_right = [peak['total_intensity'] for peak in candidate_peaks[highest_index:]]

                print(f"Intensities Left: {intensities_left}")
                print(f"Intensities Right: {intensities_right}")

                decrease_count_left = sum(intensities_left[i] >= intensities_left[i + 1] for i in range(len(intensities_left) - 1))
                decrease_count_right = sum(intensities_right[i] >= intensities_right[i + 1] for i in range(len(intensities_right) - 1))

                # Sum up total decrease and increase counts
                total_decrease_count = decrease_count_left + decrease_count_right

                print(f"Total Decrease Count: {total_decrease_count}")
                print(f"Total Peaks: {len(candidate_peaks)}")

                # Allow a soft requirement: mostly decreasing with some flexibility
                if total_decrease_count >= (len(candidate_peaks) - 1) * 0.8:
                    # Assign the charge state if a mostly decreasing pattern is found
                    for peak in candidate_peaks:
                        plot_dataframe_charge_states.at[peak.name, 'charge_state'] = charge
                    is_matched = True
                else:
                    # Mark these peaks as checked but not assigned
                    for peak in candidate_peaks:
                        plot_dataframe_charge_states.at[peak.name, 'charge_state'] = 0

            # If a valid charge state was determined, no need to check lower charges
            if is_matched:
                break

    print("Result DataFrame with Charge States:")
    print(plot_dataframe_charge_states)

    return plot_dataframe_charge_states


def choose_plot_color(input_plot_color):
    """
    Choose a color based on the input color identifier.
    """
    # Define the updated color palette
    color_data = {
        "Color Identifier": [
            "Pink1", "Pink2", "Pink3", "Pink4", "Pink5", "Pink6",
            "Gray1", "Gray2", "Gray3", "Gray4", "Gray5", "Gray6",
            "Yellow1", "Yellow2", "Yellow3", "Yellow4", "Yellow5", "Yellow6",
            "Green1", "Green2", "Green3", "Green4", "Green5", "Green6",
            "Blue1", "Blue2", "Blue3", "Blue4", "Blue5", "Blue6",
            "Red1", "Red2", "Red3", "Red4", "Red5", "Red6"
        ],
        "Hex Code": [
            "#FF36D7", "#FF1A9A", "#FF66D9", "#FF66B2", "#FF99CC", "#FFB3E6",  # Pink shades
            "#000000", "#1A1A1A", "#333333", "#4D4D4D", "#666666", "#808080",  # Grays, with Gray1 as black
            "#FFD700", "#FFC700", "#FFE700", "#FFF700", "#FFF300", "#FFEC00",  # Yellow shades
            "#2ECC40", "#28B600", "#1D9A00", "#009900", "#007700", "#005500",  # Adjusted Green shades
            "#3498DB", "#2980B9", "#1E88E5", "#007BB8", "#005EB8", "#004B9A",  # Adjusted Blue shades
            "#E74C3C", "#C0392B", "#D50000", "#A50000", "#FF5733", "#FF3333"  # Red shades
        ]
    }

    # Create the DataFrame
    colors = pd.DataFrame(color_data)

    return colors.loc[colors['Color Identifier'] == input_plot_color, 'Hex Code'].values[0]


def plot_results(plot_dataframe, output_file, plot_mass_range, plot_intensity_range, intensity_threshold, plot_color,
                 time_range, overwrite):
    """
    Plot the analyzed spectra based on the provided DataFrame.
    Display the mass values on top of each intensity bar only if the conditions are met:
    1. normalized_intensity is above the specified threshold.
    2. label is True.

    :param plot_dataframe: DataFrame containing the mass, intensity, normalized intensity, and label data to plot.
    :param output_file: Path for saving the plot.
    :param plot_mass_range: Mass range for the plot, or '0' for lower limit and '1' for upper limit to use default.
    :param plot_intensity_range: Intensity range for the plot, or '0' for lower limit and '1' for upper limit to use default.
    :param overwrite: Boolean to determine if existing plots should be overwritten.
    :param intensity_threshold: Threshold for normalized_intensity to decide which peaks to label.
    """
    plt.figure(figsize=(10, 6))

    # Create the bar plot
    plt.bar(plot_dataframe['mass'], plot_dataframe['total_intensity'], width=0.2, color=plot_color, alpha=0.7)

    # Use the base file name as the title
    file_name = os.path.basename(output_file)
    plt.title(f"{file_name}")
    plt.xlabel('Mass (m/z)')
    plt.ylabel('Intensity')

    # Adding a legend without outline or color indication
    plt.legend(handles=[mpatches.Patch(color='none', label=f"RT: {time_range[0]:.2f} - {time_range[1]:.2f}")], loc='upper right', frameon=False)  # frameon=False removes the box around the legend

    # Determine mass and intensity limits
    min_mass = plot_dataframe['mass'].min()
    max_mass = plot_dataframe['mass'].max()
    max_intensity = plot_dataframe['total_intensity'].max()

    # Set x-axis limits based on plot_mass_range or calculated values
    if plot_mass_range[0] == 0.0 and plot_mass_range[1] == 1.0:
        plt.xlim(min_mass - (max_mass - min_mass) * 0.1, max_mass + (max_mass - min_mass) * 0.1)
    else:
        plt.xlim(plot_mass_range)

    # Set y-axis limits based on plot_intensity_range or calculated values
    if plot_intensity_range[0] == 0.0 and plot_intensity_range[1] == 1.0:
        plt.ylim(0, max_intensity * 1.1)
    else:
        plt.ylim(plot_intensity_range)

    # Display mass values above each bar if conditions are met
    for index, row in plot_dataframe.iterrows():
        if row['normalized_intensity'] > intensity_threshold and row['label']:
            plt.text(
                row['mass'],  # x-coordinate (mass)
                row['total_intensity'] + (0.02 * max_intensity),  # y-coordinate slightly above the bar
                f'{row["mass"]:.4f}\nz = {row["charge_state"]}',  # Text label (mass value)
                ha='center',  # Center the text horizontally
                va='bottom',  # Position text below the y-coordinate
                fontsize=10  # Font size
            )

    # Customizing ticks
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.gca().yaxis.get_offset_text().set_fontsize(10)
    plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

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
    parser.add_argument('-charge_range',
                        help='Set a custom charge range to analyze peaks. Example: 1-5.',
                        type=str, default='1-20')
    parser.add_argument('-ppm_binning', help='Set a custom ppm range for binning masses. Example: 10.', type=str,
                        default='10')
    parser.add_argument('-ppm_charge_state', help='Set a custom ppm range for the charge state determination. Example: 10.', type=str,
                        default='10')
    parser.add_argument('-overwrite', help='If true, existing plots will be overwritten.', action='store_true')
    parser.add_argument('-full_range', help='If true, plot will span the entire time and mass range.',
                        action='store_true')
    parser.add_argument('-plot_intensity_range', help='Intensity range to use for plotting', default='0-1', type=str)
    parser.add_argument('-plot_mass_range', help='Mass range to use for plotting', default='0-1', type=str)
    parser.add_argument('-plot_color', help='Define the plot color. Example: Gray1.', default='Gray1', type=str)
    parser.add_argument('-plot_label_intensity', help='Minimum intensity of the mass labels that is still plotted. Example: 0.1', default='0.1', type=str)
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
    charge_range = [int(charge) for charge in args.charge_range.split('-')]
    ppm_binning = float(args.ppm_binning)
    label_intensity_threshold = float(args.plot_label_intensity)

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
            plot_dataframe = generate_plot_dataframe_with_charge_states(generate_plot_dataframe(analyzed_spectra, ppm_binning),
                                                                        charge_range , float(args.ppm_charge_state), label_intensity_threshold)

            sys.stdout.write(f"\tFound {len(plot_dataframe)} unique masses for plotting.\n")
            plot_results(plot_dataframe, os.path.splitext(input_file)[0], plot_mass_range, plot_intensity_range,
                         label_intensity_threshold, choose_plot_color(args.plot_color), time_range, args.overwrite)

    sys.stdout.write(f"{''.join(['=' for _ in range(20)])}\n")


if __name__ == '__main__':
    main()
