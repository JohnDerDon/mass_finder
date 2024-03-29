# -*- coding: utf-8 -*-
"""
Parse through mzXML files to find specific mass patterns that match specific
chemical formulas.

(C) Mathijs Mabesoone, ETH Zurich
February 2022
"""
from pyteomics import mzxml
import os
import sys
import argparse
import multiprocess as mp
import pandas as pd
import matplotlib.pyplot as plt
from numpy import log10
from math import ceil, floor
from collections import defaultdict
import numpy as np
from platform import system
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from pyteomics import mass as pymass

def analyze_mass_spec(spectrum, mass_range, accuracy, formulas_with_charge, min_intensity):
    # Convert arrays to numpy arrays
    if spectrum.get('msLevel', 0) == 1:  # Check if the spectrum is MS1
        mz_array = np.array(spectrum.get('m/z array', []))
        intensity_array = np.array(spectrum.get('intensity array', []))
    else:
        return None

    # Check if arrays are empty
    if mz_array.size == 0 or intensity_array.size == 0:
        return None

    matching_masses = []

    for index, experimental_mass in enumerate(mz_array):
        if mass_range[0] <= experimental_mass <= mass_range[1] and intensity_array[index] >= min_intensity:
                for formula in formulas_with_charge:
                    formula_masses = formulas_with_charge[formula]
                    for charge_mass in formula_masses:
                        charge, formula_mass = charge_mass
                        if abs(formula_mass - experimental_mass) < accuracy * experimental_mass:
                            matching_masses.append({
                                'index': index,
                                'intensity': round(intensity_array[index], 0),
                                'experimental_mass': experimental_mass,
                                'formula': formula,
                                'theoretical_mass': formula_mass,
                                'charge_state': charge,
                                'parent_mass': formula_mass * charge - charge * 1.0073
                            })
                            # print(formula, formula_mass, charge, experimental_mass)
                        # Assuming the formulas_with_charge dictionary is organized from low to high mass

    return (float(spectrum.get('retentionTime', 0)), matching_masses) if len(matching_masses) > 0 else None


def construct_element_dictionary(element_string):
    # Construct the element dictionary
    if element_string is None:
        return None
    element_dictionary = {}
    elements = element_string.split('_')
    for element in elements:
        parts = element.split('-')
        assert len(parts) == 3
        min_count = int(parts[0])
        max_count = int(parts[2])
        identifier = parts[1]
        # check if there is a custom mass
        if ':' in identifier:
            identifier_parts = identifier.split(':')
            mass = float(identifier_parts[1])
            identifier = identifier_parts[0]
        # check if there is a peptide input sequence
        elif all(char in "ACDEFGHIKLMNPQRSTVWY" for char in identifier) and len(identifier) > 1:
            peptide = identifier
            # peptide mass minus H2O for concatenation of different peptide stretches. 1 H2O must be added in the command line.
            mass = pymass.calculate_mass(sequence=peptide) - pymass.calculate_mass(formula='H2O')
        # get all atomic masses
        else:
            # get the mass of the chemical formula
            mass = pymass.calculate_mass(formula=identifier)
        element_dictionary[identifier] = [min_count, max_count, round(mass, 4)]
    return element_dictionary


def generate_formulas(element_string):
    formulas = {}
    element_dict = construct_element_dictionary(element_string)
    elements = list(element_dict.keys())

    def backtrack(formula, current_element, mass):
        """Recursively generate all possible formulas"""
        if current_element == len(elements):
            formulas[formula] = mass
            return

        element = elements[current_element]
        min_count, max_count, element_mass = element_dict[element]

        for count in range(min_count, max_count + 1):
            updated_formula = f"{formula}{element}{count}"
            updated_mass = mass + (element_mass * count)
            # If the count is non-zero, proceed recursively
            if count > 0:
                backtrack(updated_formula, current_element + 1, updated_mass)
            else:
                # If the count is zero, proceed without adding the element
                backtrack(formula, current_element + 1, mass)

    backtrack("", 0, 0.0)
    # Sort the formulas dictionary with a lambda function that sorts by the mass
    formulas = {formula: mass for formula, mass in sorted(formulas.items(), key=lambda item: item[1])}

    return formulas


def generate_formula_with_charge(formulas, mass_range, monoisotopic):
    formulas_with_charge = {}
# find maximum and minimum charge states for each formula
    for formula, mass in formulas.items():
        min_charge, max_charge = calculate_charge_range(mass, mass_range)
        # calculate the number of the most abundant 13C isotope peak, empirically determined to change at 1500 Da
        isotope_peak_number = int(mass / 1500)
        print(formula, mass, min_charge, max_charge)

        for charge in range(min_charge, max_charge + 1):
            if monoisotopic:
                charge_state_mass = round((mass + (charge * 1.0073)) / charge, 4)
            else:
                # calculate the mass of the most abundant 13C isotope peak, empirically determined to change at 1500 Da
                charge_state_mass = round((mass + (charge * 1.0073) + (1.003354835 * isotope_peak_number)) / charge, 4)
            if formula in formulas_with_charge:
                formulas_with_charge[formula].append([charge, charge_state_mass])
            else:
                formulas_with_charge[formula] = [[charge, charge_state_mass]]
    print(formulas_with_charge)

    return formulas_with_charge


def calculate_charge_range(mass, mass_range):
    min_charge = max(1, int((mass + mass_range[1]) / mass_range[1]))
    max_charge = max(1, int((mass + mass_range[0]) / mass_range[0])) - 1

    return min_charge, max_charge


def append_suffix_to_file(file, overwrite):
    # Check if files already exist and if so, append a suffix to the file name
    file = os.path.splitext(file)[0]
    if overwrite:
        return file
    if any([os.path.isfile(file + extension) for extension in ['.svg', '.png', '_analyzed.txt']]):
        suffix = 1
        while any([os.path.isfile(f"{file}_{suffix}{extension}") for extension in ['.svg', '.png', '_analyzed.txt']]):
            suffix += 1
        return f"{file}_{suffix}"
    else:
        return file


def plot_results_in_2D(analyzed_spectra, output_file, time_range, mass_range, overwrite, full_range, min_intensity, group_identifiers):
    # Plot the analyzed spectra in a single graph
    log_min_intensity = log10(min_intensity)
    plot_list = list()
    for spectrum in [spectrum for spectrum in analyzed_spectra if spectrum is not None]:
        time = spectrum[0]
        for peak in spectrum[1:]:
            peaks = [value for value in peak]
            for peak in peaks:
                plot_list.append([peak['experimental_mass'], time, log10(peak['intensity']),
                                  peak['formula'], peak['parent_mass']])
    plot_list = pd.DataFrame(plot_list, columns=['experimental_mass', 'time', 'intensity', 'formula', 'parent_mass'])
    plot_list = plot_list.sort_values(by='intensity', ascending=True, ignore_index=True)

    # Check if plot_list is empty
    if plot_list.empty:
        sys.stdout.write(f"Nothing to plot. The plot list is empty.\n")
        return

    # Create a DataFrame with identifiers and their corresponding parent masses sorted by parent mass
    sorted_df = plot_list[['formula', 'parent_mass']].sort_values(by='parent_mass')
    # Get the sorted unique identifiers as a list
    sorted_unique_identifiers = sorted_df['formula'].drop_duplicates().tolist()

    num_identifiers = len(sorted_unique_identifiers)

    # Generate a custom colormap with a varying number of colors
    # check if the number of identifiers is divisible by the number of group identifiers
    if num_identifiers % group_identifiers != 0:
        raise ValueError('The number of identifiers must be divisible by the number of similar groups')
    # calculate the number of groups
    num_groups = num_identifiers // group_identifiers

    colors = np.array([]).reshape(0, 4)
    # Generate a list of colors using the tab20 colormap based on the number of unique identifiers
    for i in range(num_groups):
        for j in range(group_identifiers):
            color_distribution = ((i + 1 / group_identifiers) / num_groups + j / (
                        2 * (num_groups + group_identifiers)))
            colors = np.append(colors, [plt.cm.rainbow(color_distribution)], axis=0)

    # Define the plot
    plt.figure(figsize=(12, 10))
    plt.title(os.path.splitext(os.path.basename(output_file))[0], fontsize=22)
    for i, identifier in enumerate(sorted_unique_identifiers):
        indices = plot_list.index[plot_list['formula'] == identifier].tolist()  # Get indices where identifier matches
        color = colors[i]
        for j in indices:
            alpha = (plot_list.at[j, 'intensity'] - log_min_intensity) / (plot_list['intensity'].max() - log_min_intensity)  # Normalize z value for shading
            plt.scatter(plot_list.at[j, 'time'], plot_list.at[j, 'experimental_mass'],
                        marker='.', edgecolors='none', color=color, alpha=alpha, label=f'{identifier}')

    # label specifications
    plt.xlabel('Time (min)', fontsize=18)
    plt.ylabel('m/z', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # check if full_range is true, otherwise adapt range
    if full_range:
        plt.xlim((min(time_range), max(time_range)))
        plt.ylim((min(mass_range), max(mass_range)))
    else:
        plt.xlim((0.9 * min(plot_list['time']), 1.1 * max(plot_list['time'])))
        plt.ylim((0.9 * min(plot_list['experimental_mass']), 1.1 * max(plot_list['experimental_mass'])))

    plt.grid(which='both', alpha=0.3)

    # Create a ScalarMappable object for the intensity values
    alpha_sm = plt.cm.ScalarMappable(cmap=plt.cm.gray_r, norm=plt.Normalize(vmin=log_min_intensity, vmax=plot_list['intensity'].max()))
    alpha_sm.set_array([])  # Setting an empty array

    # Add a color bar representing intensity values
    cbar = plt.colorbar(alpha_sm, shrink=0.5, ax=plt.gca())
    cbar.ax.set_ylabel('log(intensity)', rotation=270, labelpad=20, fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Set ticks on the colorbar with increments of 1
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Create legend with custom font color
    patches = []
    for i, identifier in enumerate(sorted_unique_identifiers):
        patches.append(mpatches.Patch(color=colors[i], label=identifier))
    plt.legend(handles=patches, fontsize='large')

    # Save the plot
    if not overwrite:
        if any([os.path.isfile(output_file + '.svg'), os.path.isfile(output_file + '.png')]):
            suffix = 1
            while any([os.path.isfile(output_file + '_' + str(suffix) + '.svg'),
                       os.path.isfile(output_file + '_' + str(suffix) + '.png')]):
                suffix += 1
            output_file = f"{output_file}_{suffix}"
    plt.savefig(output_file + '.svg', transparent=True, dpi=300)
    plt.savefig(output_file + '.png', transparent=True, dpi=300)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Check an xml files in a folder for a specific isotope pattern.')
    parser.add_argument('-input', help='Input folder or file')
    parser.add_argument('-threads', help='Number of threads to use', type=int, default=1)
    parser.add_argument('-min_intensity', help='Minimal intensity of main peak to report. Default = 1e4.', type=float, default=1e4)
    parser.add_argument('-elements',
                        help='Define the boundaries for elemental composition. Input format [minimal number]-[element]-[maximal number]_[next element]. E.g. "2-C-10_2-N-5_0-H-20]". For custom masses, such as between 1 and 2 phenols, add 1-phenol:94.0419-2.',
                        type=str, default=None)
    parser.add_argument('-accuracy',
                        help='Tolerance of relative mass difference between measured and predicted masses. Default = 5e-6.',
                        type=float, default=5e-6)
    parser.add_argument('-time_range',
                        help='Set a custom time range to analyze in the mass spec data. Example: for 3-10 minutes, enter 3-10.',
                        type=str, default='0-1000')
    parser.add_argument('-mass_range',
                        help='Set a custom mass range to analyze in the mass spec data. Example: for m/z 200-600 , enter 200-600.',
                        type=str, default='200-2000')
    parser.add_argument('-output_folder',
                        help='Specify a specific output folder. If not specified, the output will be in the same folder as the mzxml files.',
                        type=str)
    parser.add_argument('-output_prefix', help='Prefix to the output files', default='', type=str)
    parser.add_argument('-overwrite',
                        help='If overwrite is True, the data saved from previous runs will be overwritten.',
                        action='store_true')
    parser.add_argument('-full_range',
                        help='If full_range is True, the output plot will span the entire time and mass range. Useful for comparing samples, but less ideal to check a single file. Default: False',
                        action='store_true')
    parser.add_argument('-monoisotopic',
                        help='If full_range is True, the output plot will span the entire time and mass range. Useful for comparing samples, but less ideal to check a single file. Default: False',
                        action='store_true')
    parser.add_argument('-plot_time_range', help='Time range to use for plotting', default='0-30', type=str)
    parser.add_argument('-plot_mass_range', help='Mass range to use for plotting', default='200-2000', type=str)
    parser.add_argument('-plot_group_identifiers', help='Group similar identifiers in the plot with similar colors', default=1, type=int)

    args = parser.parse_args()

    # Print boundary
    sys.stdout.write(f"{''.join(['=' for _ in range(20)])}\n")
    # Check if the input is a directory or file and make a file list
    if os.path.isdir(args.input) and not os.path.isfile(args.input):
        files = [os.path.join(args.input, file) for file in os.listdir(args.input) if '.mzxml' in file.lower()]
        sys.stdout.write(f"Detected {len(files)} .MZxml files to analyze in {args.input}:\n")
        sys.stdout.write('\t' + '\n\t'.join(files) + '\n')
    elif os.path.isfile(args.input) and '.mzxml' in args.input.lower():
        files = [args.input]
        sys.stdout.write(f"Analyzing single file: {args.input}.\n")
    else:
        sys.stdout.write(f"Did not detect any MZxml files in {args.input}. Terminating...\n")
        return
    if args.elements is None:
        sys.stdout.write(f"No elements string received. Exiting.\n")
        return
    if args.full_range and args.plot_mass_range == '200-2000':
        operating_system = system()
        if operating_system == 'Windows':
            reply = ''
        else:
            sys.stdout.write(f"Running on Linux. Using standard mass_range: 200-2000.\n")
            reply = 'n'
        while reply.lower() not in ['y', 'n']:
            reply = input(
                'No mass range specified? The original range is not saved in the mzXML files. Do you want to specify your own instead of using the default 200-2000? [y/n]:\n')
        if reply.lower == 'y':
            args.plot_mass_range = input("Input your desired mass range:\n")

    # Analyze for each file all spectra in parallel. Write output of each file to a txt
    pool = mp.Pool(args.threads)
    mass_range = [float(mass) for mass in args.mass_range.split('-')]
    time_range = [float(time) for time in args.time_range.split('-')]
    formulas_with_charge = generate_formula_with_charge(generate_formulas(args.elements), mass_range, args.monoisotopic)

    nl = '\n\t\t'  # new line for f-strings

    for file in files:
        sys.stdout.write(f"Started parsing {file}\n")
        data = mzxml.MzXML(file, use_index=True)
        min_index, max_index = [int(data.time[float(time)]['id']) for time in time_range]
        analyzed_spectra = pool.starmap(analyze_mass_spec, [(data.get_by_index(int(index) - 1), mass_range,
                                                             args.accuracy, formulas_with_charge, args.min_intensity)
                                                            for index in range(min_index, max_index)])
        analyzed_spectra = [spectrum for spectrum in analyzed_spectra if spectrum is not None]
        sys.stdout.write(
            f"\tFound {sum([len(spectrum[1]) for spectrum in analyzed_spectra if spectrum != None])} masses matching the pattern in {file}\n")

        if args.output_folder is not None:
            if not os.path.isdir(args.output_folder):
                sys.stdout.write(
                    f"WARNING: {args.output_folder} is not a valid folder path. Saving files in {os.path.dirname(file)} instead.\n")
            else:
                file = os.path.join(args.output_folder, os.path.basename(file))
        file = append_suffix_to_file(os.path.join(os.path.dirname(file), str(args.output_prefix) + os.path.basename(file)),
                                     args.overwrite)
        with open(os.path.splitext(file)[0] + '_analyzed.txt', 'w') as output_file:
            output_file.write(
                f"Checking for compounds with formulas in range {args.elements}.\nChecking in time range {time_range} and mass range {mass_range}.\n\n")
            output_file.write(
                f"Mass range; {mass_range}\nTime range: {time_range}\n\n")
            output_file.write(
                f"Found {sum([len(spectrum[1]) for spectrum in analyzed_spectra if spectrum != None])} matching masses in {file}\n")

            for retention_time, spectrum in analyzed_spectra:
                if retention_time is None:
                    continue
                output_file.write(f"Found matching mass at {retention_time} min:\n")
                for peak in spectrum:
                    output_file.write(
                        f"\tExperimental Mass: {peak['experimental_mass']}\tIntensity: {peak['intensity']}\tFormula: {peak['formula']}\tTheoretical mass: {peak['theoretical_mass']}\tCharge state: {peak['charge_state']}" + "\n")
        if args.full_range:
            plot_time_range = [round(float(data.time[float(time)]['retentionTime']),2) for time in args.plot_time_range.split('-')]
            plot_mass_range = [float(mass) for mass in args.plot_mass_range.split('-')]
            plot_results_in_2D(analyzed_spectra, os.path.splitext(file)[0], plot_time_range, plot_mass_range,
                               args.overwrite, args.full_range, args.min_intensity, args.plot_group_identifiers)
        else:
            plot_time_range = [round(float(data.time[float(time)]['retentionTime']), 2) for time in args.plot_time_range.split('-')]
            plot_mass_range = [float(mass) for mass in args.plot_mass_range.split('-')]
            plot_results_in_2D(analyzed_spectra, os.path.splitext(file)[0], plot_time_range, plot_mass_range,
                               args.overwrite, args.full_range, args.min_intensity, args.plot_group_identifiers)
    pool.close()


if __name__ == '__main__':
    main()
