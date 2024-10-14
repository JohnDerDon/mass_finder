# -*- coding: utf-8 -*-
"""
Parse through mzXML files to find specific compounds based on their mass and intensity.

(C) Johannes Eckert, Mathijs Mabesoone, ETH Zurich
April 2024
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
from scipy import constants
from textwrap import wrap


def analyze_mass_spec(spectrum, mass_range, accuracy, formulas_with_charge, min_intensity):
    """
    Analyzes a mass spectrum to find peaks matching theoretical masses within a specified accuracy range.

    Parameters:
    - spectrum (dict): Dictionary containing 'm/z array' and 'intensity array' from mass spectrometry data.
    - mass_range (tuple): Tuple specifying the minimum and maximum m/z values to consider.
    - accuracy (float): Accuracy threshold for matching experimental masses to theoretical masses.
    - formulas_with_charge (dict): Dictionary where keys are molecular formulas and values are lists of tuples containing charge, theoretical mass, and isotope peak number.
    - min_intensity (float): Minimum intensity threshold to consider a peak.

    Returns:
    - tuple: Retention time and list of matching masses with details, or None if no matches are found.
    """

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

    # calculate the mass of a proton in dalton
    mass_proton_kilo = constants.physical_constants['proton mass'][0]
    kilo_in_dalton = constants.physical_constants['atomic mass unit-kilogram relationship'][0]
    mass_proton = mass_proton_kilo / kilo_in_dalton

    # Iterate through each experimental mass in the spectrum
    # and check if it matches any of the theoretical masses
    # within the specified mass range and intensity threshold
    # If a match is found, add the peak to the list of matching masses
    # along with the formula, theoretical mass, charge state, and isotope peak number
    for index, experimental_mass in enumerate(mz_array):
        if mass_range[0] <= experimental_mass <= mass_range[1] and intensity_array[index] >= min_intensity:
                for formula in formulas_with_charge:
                    formula_masses = formulas_with_charge[formula]
                    for charge_mass in formula_masses:
                        charge, formula_mass, isotope_peak_number = charge_mass
                        if abs(formula_mass - experimental_mass) < accuracy * experimental_mass:
                            matching_masses.append({
                                'index': index,
                                'intensity': round(intensity_array[index], 0),
                                'experimental_mass': experimental_mass,
                                'formula': formula,
                                'theoretical_mass': formula_mass,
                                'charge_state': charge,
                                'isotope_peak_number': isotope_peak_number,
                                'parent_mass': formula_mass * charge - charge * mass_proton
                            })
                        # Assuming the formulas_with_charge dictionary is organized from low to high mass
    if len(matching_masses) > 0:
        analyzed_spectra = (float(spectrum.get('retentionTime', 0)), matching_masses)
    else:
        return None

    return analyzed_spectra


def construct_element_dictionary(element_string):
    """
    Construct a dictionary of elements from a given element string.

    Parameters:
    element_string (str): The input string defining elements and their counts.

    Returns:
    dict: A dictionary with element identifiers as keys and their counts, mass, and isotope count as values.
    """

    # Construct the element dictionary
    if element_string is None:
        return None
    # Split the element string into individual elements
    # and extract the minimum and maximum counts, the identifier, and the mass
    element_dictionary = {}
    elements = element_string.split(';')
    relative_abundance_13C = pymass.nist_mass['C'][13][1]  # Approximately 1.07% in nature from pymass
    for element in elements:
        parts = element.split('_')
        assert len(parts) == 3
        # Check if the minimum count is set to 'X', indicating that the element can only be added once
        if parts[0] != 'X':
            min_count = int(parts[0])
        else:
            min_count = False
        max_count = int(parts[2])
        identifier = parts[1]
        # check if there is a custom mass
        if ':' in identifier:
            identifier_parts = identifier.split(':')
            mass = float(identifier_parts[1])
            identifier = identifier_parts[0]
            isotope_count = mass/1500  # approximate value for isotopologues in custom mass
        # check if there is a peptide input sequence
        elif all(char in "ACDEFGHIKLMNPQRSTVWY" for char in identifier) and len(identifier) > 1:
            peptide = identifier
            # peptide mass minus H2O for concatenation of different peptide stretches. 1 H2O must be added in the command line.
            mass = pymass.calculate_mass(sequence=peptide) - pymass.calculate_mass(formula='H2O')
            # get the 13C isotope count of the peptide
            num_carbon_atoms = pymass.Composition(identifier).get('C', 0)
            isotope_count = relative_abundance_13C * num_carbon_atoms
        # get all atomic masses
        else:
            # get the mass of the chemical formula
            mass = pymass.calculate_mass(formula=identifier)
            # get the 13C isotope count of the chemical formula
            num_carbon_atoms = pymass.Composition(identifier).get('C', 0)
            isotope_count = relative_abundance_13C * num_carbon_atoms
        element_dictionary[identifier] = [min_count, max_count, round(mass, 4), round(isotope_count, 4)]
    return element_dictionary


def generate_formulas(element_string):
    """
    Generate all possible formulas from the given element string.

    Parameters:
    element_string (str): The input string defining elements and their counts.

    Returns:
    dict: A dictionary of formulas with their masses and isotope counts.
    """

    formulas = {}
    element_dict = construct_element_dictionary(element_string)
    elements = list(element_dict.keys())

    # check for single use elements in the element dictionary
    single_use_elements = {element for element, (min_count, _, _, _) in element_dict.items() if min_count is False}

    # Recursively generate all possible formulas

    def backtrack(formula, current_element, mass, isotope_count, used_single_use_element):
        """Recursively generate all possible formulas"""
        if current_element == len(elements):
            formulas[formula] = (mass, isotope_count)
            return

        element = elements[current_element]
        min_count, max_count, element_mass, element_isotope_count = element_dict[element]

        # check if the element is a single use element
        # if it is, check if it has already been used
        # if it has not been used, add the element to the formula
        if element in single_use_elements:
            if not used_single_use_element:
                if max_count != 0:
                    updated_formula = f"{formula}{element}({max_count})"
                    updated_mass = mass + (element_mass * max_count)
                    updated_isotope_count = isotope_count + (element_isotope_count * max_count)
                    backtrack(updated_formula, current_element + 1, updated_mass, updated_isotope_count, element)
            backtrack(formula, current_element + 1, mass, isotope_count, used_single_use_element)
        else:
            for count in range(min_count, max_count + 1):
                updated_formula = f"{formula}{element}({count})"
                updated_mass = mass + (element_mass * count)
                updated_isotope_count = isotope_count + (element_isotope_count * count)
                # If the count is non-zero, proceed recursively
                if count != 0:
                    backtrack(updated_formula, current_element + 1, updated_mass, updated_isotope_count,
                              used_single_use_element)
                else:
                    # If the count is zero, proceed without adding the element
                    backtrack(formula, current_element + 1, mass, isotope_count,
                              used_single_use_element)

    backtrack("", 0, 0.0, 0.0, None)
    formulas = {formula: (mass, isotope_count) for formula, (mass, isotope_count) in
                sorted(formulas.items(), key=lambda item: item[1][0])}

    return formulas


def generate_formula_with_charge(formulas, mass_range, monoisotopic):
    """
    Calculate the charge states and corresponding masses for each formula within a given mass range.

    Parameters:
    formulas (dict): Dictionary of formulas with their masses and isotope counts.
    mass_range (list): List with minimum and maximum mass range values.
    monoisotopic (bool): Flag to indicate whether to use monoisotopic masses.

    Returns:
    dict: A dictionary with formulas as keys and lists of charge states, masses, and isotope peak numbers as values.
    """

    formulas_with_charge = {}

    # calculate the masses of a proton and a neutron for the isotope peak calculation
    mass_proton_kilo = constants.physical_constants['proton mass'][0]
    kilo_in_dalton = constants.physical_constants['atomic mass unit-kilogram relationship'][0]
    mass_proton = mass_proton_kilo / kilo_in_dalton
    mass_neutron = pymass.nist_mass['C'][13][0] - pymass.nist_mass['C'][12][0]

    # find maximum and minimum charge states for each formula
    for formula, (mass, isotope_count) in formulas.items():
        min_charge, max_charge = calculate_charge_range(mass, mass_range)
        # calculate the number of the most abundant 13C isotope peak with the values of the isotope count
        isotope_peak_number = round(isotope_count)

        for charge in range(min_charge, max_charge + 1):
            # check if monoisotopic is set to True
            if monoisotopic:
                charge_state_mass = round((mass + (charge * mass_proton)) / charge, 4)
            else:
                # calculate the mass of the most abundant 13C isotope peak, empirically determined to change at 1500 Da
                charge_state_mass = round((mass + (charge * mass_proton) + (mass_neutron * isotope_peak_number)) / charge, 4)
            if formula in formulas_with_charge:
                formulas_with_charge[formula].append([charge, charge_state_mass, isotope_peak_number])
            else:
                formulas_with_charge[formula] = [[charge, charge_state_mass, isotope_peak_number]]

    return formulas_with_charge


def calculate_charge_range(mass, mass_range):
    """
    Calculate the minimum and maximum charge states for a given mass based on the mass range.

    Parameters:
    mass (float): The mass of the formula.
    mass_range (list): List with minimum and maximum mass range values.

    Returns:
    tuple: Minimum and maximum charge states.
    """

    # Calculate the minimum and maximum charge states for a given mass based on the mass range
    min_charge = max(1, int((mass + mass_range[1]) / mass_range[1]))
    max_charge = max(1, int((mass + mass_range[0]) / mass_range[0])) - 1

    return min_charge, max_charge


def append_suffix_to_file(file, overwrite):
    # Check if files already exist and if so, append a suffix to the file name
    file = os.path.splitext(file)[0]

    #  check if the file should be overwritten
    if overwrite:
        return file
    if any([os.path.isfile(file + extension) for extension in ['.svg', '.png', '_analyzed.txt']]):
        suffix = 1
        while any([os.path.isfile(f"{file}_{suffix}{extension}") for extension in ['.svg', '.png', '_analyzed.txt']]):
            suffix += 1
        return f"{file}_{suffix}"
    else:
        return file


def generate_plot_list(analyzed_spectra):

    # Generate a DataFrame with the experimental mass, time, intensity, log10(intensity), formula, and parent mass
    # for each peak in the analyzed spectra
    # The DataFrame is sorted by time in ascending order
    plot_list = list()
    for spectrum in [spectrum for spectrum in analyzed_spectra if spectrum is not None]:
        time = spectrum[0]
        for peak in spectrum[1:]:
            peaks = [value for value in peak]
            for peak in peaks:
                plot_list.append([peak['experimental_mass'], time, peak['intensity'], log10(peak['intensity']),
                                  peak['formula'], peak['parent_mass']])
    plot_list = pd.DataFrame(plot_list, columns=['experimental_mass', 'time', 'intensity', 'intensity_log10',
                                                 'formula', 'parent_mass'])
    plot_list = plot_list.sort_values(by='time', ascending=True, ignore_index=True)

    if plot_list.empty:
        sys.stdout.write(f"Nothing to plot. The plot list is empty.\n")
        return None

    return plot_list


def relative_abundances(plot_list):
    """
    Calculate the relative abundances of each formula based on the sum of intensities.

    Parameters:
        df (DataFrame): DataFrame with 'formula' and 'intensity' columns.

    Returns:
        dict: Dictionary containing the relative abundances of each formula.
    """
    # check if plot_list is type NoneType
    if plot_list is None:
        return None

    # Group by 'formula' and sum the 'intensity' within each group
    sum_intensities = plot_list.groupby('formula')['intensity'].sum()

    # Calculate the total intensity
    total_intensity = sum_intensities.sum()

    # Calculate the relative abundances for each formula
    relative_abundances = {formula: intensity / total_intensity for formula, intensity in sum_intensities.items()}

    return relative_abundances



def plot_results(analyzed_spectra, output_file, time_range, mass_range, overwrite, full_range, min_intensity):
    # Plot the analyzed spectra in a single graph

    # calculate the logarithm of the minimal intensity
    log_min_intensity = log10(min_intensity)
    # calculate the plot list
    plot_list = generate_plot_list(analyzed_spectra)

    # Create a DataFrame with identifiers and their corresponding parent masses sorted by parent mass
    sorted_df = plot_list[['formula', 'parent_mass']].sort_values(by='parent_mass')
    # Get the sorted unique identifiers as a list
    sorted_unique_identifiers = sorted_df['formula'].drop_duplicates().tolist()
    num_identifiers = len(sorted_unique_identifiers)



    suffix = 0
    if not overwrite:
        if os.path.isfile(output_file + '.svg'):
            suffix = 1
            while os.path.isfile(output_file + '_' + str(suffix) + '.svg'):
                suffix += 1

    # Plot the results
    plot_results_subplots(plot_list, output_file, time_range, mass_range, full_range, sorted_unique_identifiers,
                          inverted_colors, log_min_intensity, suffix, min_intensity)


def plot_results_subplots(plot_list, output_file, time_range, mass_range,
                          full_range, sorted_unique_identifiers, colors, log_min_intensity, suffix, min_intensity):
    # Define the plot
    fig = plt.figure(figsize=(16, 16))
    # Define the title of the plot
    fig.suptitle(os.path.splitext(os.path.basename(output_file))[0], fontsize=22)

    # Create a grid for the subplots
    gs = fig.add_gridspec(2, 5, width_ratios=[2, 1, 12, 2, 2], height_ratios=[1, 1])
    ax_abundance = fig.add_subplot(gs[0:, 0])
    ax_XIC = fig.add_subplot(gs[0, 2])
    ax_scatter = fig.add_subplot(gs[1, 2])
    ax_colorbar = fig.add_subplot(gs[1, 3])
    ax_colorbar.axis('off')
    ax_legend = fig.add_subplot(gs[0, 3:])
    ax_legend.axis('off')

    # Plot the stacked bar
    plot_stacked_bar(ax_abundance, plot_list, sorted_unique_identifiers, colors, relative_abundances)

    # Plot the XIC
    plot_XIC(ax_XIC, plot_list, time_range, full_range, sorted_unique_identifiers, colors, min_intensity, ax_legend)

    # Plot the scatter plot
    plot_scatter(ax_scatter, plot_list, time_range, mass_range, full_range, sorted_unique_identifiers,
                 colors, log_min_intensity, ax_colorbar)

    # Save the plot
    if suffix != 0:
        output_file = f"{output_file}_{suffix}"
    plt.savefig(output_file + '.svg', transparent=True, dpi=300)
    plt.close()


def plot_stacked_bar(ax_abundance, plot_list, sorted_unique_identifiers, colors, relative_abundances):
    # Define the abundance plot on the abundance axis
    abundances = relative_abundances(plot_list)

    # Combine all intensities into a single array for stacking
    stacked_intensities = [abundances[sorted_unique_identifiers[0]]]
    for i in range(len(sorted_unique_identifiers) - 1):
        formula2 = sorted_unique_identifiers[i + 1]
        stacked_intensity = stacked_intensities[i] + abundances[formula2]
        stacked_intensities.append(stacked_intensity)

    # Reverse the order of the intensities & the colors for the stacked bar plot
    stacked_intensities = stacked_intensities[::-1]
    colors = colors[::-1]

    # Plot a stacked bar in reverse order
    ax_abundance.bar(0, stacked_intensities, align='center', color=colors)

    ax_abundance.set_xticks([])  # Hide x-axis ticks
    ax_abundance.tick_params(axis='y', which='major', labelsize=14)
    ax_abundance.set_ylabel('Relative Abundance', fontsize=18)


def plot_scatter(ax_scatter, plot_list, time_range, mass_range,
                 full_range, sorted_unique_identifiers, colors, log_min_intensity, ax_colorbar):

    # plot the scatter plot on the main axes
    for i, identifier in enumerate(sorted_unique_identifiers):
        indices = plot_list.index[plot_list['formula'] == identifier].tolist()  # Get indices where identifier matches
        color = colors[i]
        for j in indices:
            alpha = ((plot_list.at[j, 'intensity_log10'] - log_min_intensity) /
                     (plot_list['intensity_log10'].max() - log_min_intensity))  # Normalize z value for shading
            ax_scatter.scatter(plot_list.at[j, 'time'], plot_list.at[j, 'experimental_mass'],
                            marker='.', edgecolors='none', color=color, alpha=alpha, label=f'{identifier}')

    # label specifications
    ax_scatter.set_xlabel('Time (min)', fontsize=18)
    ax_scatter.set_ylabel('m/z', fontsize=18)
    ax_scatter.tick_params(axis='both', which='major', labelsize=14)

    # check if full_range is true, otherwise adapt range
    if full_range:
        ax_scatter.set_xlim((min(time_range), max(time_range)))
        ax_scatter.set_ylim((min(mass_range), max(mass_range)))
    else:
        ax_scatter.set_xlim((0.9 * min(plot_list['time']), 1.1 * max(plot_list['time'])))
        ax_scatter.set_ylim((0.9 * min(plot_list['experimental_mass']), 1.1 * max(plot_list['experimental_mass'])))

    ax_scatter.grid(which='both', alpha=0.3)

    # Create a ScalarMappable object for the intensity values
    alpha_sm = plt.cm.ScalarMappable(cmap=plt.cm.gray_r, norm=plt.Normalize(vmin=log_min_intensity, vmax=plot_list['intensity_log10'].max()))
    alpha_sm.set_array([])  # Setting an empty array

    # Add a color bar representing intensity values shifted to the left of the plot
    cbar = plt.colorbar(alpha_sm, ax=ax_colorbar,  orientation='vertical')
    cbar.ax.set_ylabel('log(intensity)', rotation=270, labelpad=20, fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Set ticks on the colorbar with increments of 1
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


def plot_XIC(ax_XIC, plot_list, time_range, full_range,
                 sorted_unique_identifiers, colors, min_intensity, ax_legend):

    max_values = plot_list.groupby('formula')['intensity'].max()  # Get the maximum intensity for each identifier
    max_values_sorted = max_values.sort_values(ascending=False)
    # Create a zorder for the identifiers based on their maximum intensity
    identifier_zorder = {identifier: i for i, identifier in enumerate(max_values_sorted.index)}

    # Define the XIC main plot
    # Create a dictionary to store intensity values for each time point and identifier
    time_intensity_map = defaultdict(dict)

    # Iterate through each identifier
    for i, identifier in enumerate(sorted_unique_identifiers):
        indices = plot_list.index[plot_list['formula'] == identifier].tolist()  # Get indices where identifier matches
        color = colors[i]

        # Iterate through each index corresponding to the current identifier
        for j in indices:
            time_point = plot_list.at[j, 'time']
            intensity = plot_list.at[j, 'intensity']

            # If the time point already exists for the current identifier, add intensity to existing value
            if time_point in time_intensity_map[identifier]:
                time_intensity_map[identifier][time_point] += intensity
            else:
                time_intensity_map[identifier][time_point] = intensity

        # from the dictionary, create lists of times and intensities & add the start and end points
        # added timepoints are necessary to create a continuous line plot
        # 2nd & 3rd insertion to ensure that the line plot has a peak when having low intensity samples
        times = list(time_intensity_map[identifier].keys())
        intensities = list(time_intensity_map[identifier].values())
        times.insert(0, time_range[0])
        times.insert(1, (times[1]-0.01))
        times.append(times[-1] + 0.01)
        times.append(time_range[1])
        intensities.insert(0, min_intensity)
        intensities.insert(1, min_intensity)
        intensities.append(min_intensity)
        intensities.append(min_intensity)

        ax_XIC.plot(times, intensities, color=color, linewidth=1.5, label=f'{identifier}', zorder=identifier_zorder[identifier])

    # label specifications
    ax_XIC.set_xlabel('Time (min)', fontsize=18)
    ax_XIC.set_ylabel('Intensity', fontsize=18)
    ax_XIC.tick_params(axis='both', which='major', labelsize=14)

    # check if full_range is true, otherwise adapt range
    if full_range:
        ax_XIC.set_xlim((min(time_range), max(time_range)))
    else:
        ax_XIC.set_xlim((0.9 * min(plot_list['time']), 1.1 * max(plot_list['time'])))

    ax_XIC.grid(which='both', alpha=0.3)

    # Create legend with custom font color
    patches = []
    for i, identifier in enumerate(sorted_unique_identifiers):
        patches.append(mpatches.Patch(color=colors[i], label='\n'.join(wrap(identifier, 25))))
    ax_legend.legend(handles=patches, fontsize='large', loc='upper left')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Check XML files in a folder for a specific isotope pattern.')
    parser.add_argument('-input', help='Input folder or file')
    parser.add_argument('-threads', help='Number of threads to use', type=int, default=1)
    parser.add_argument('-min_intensity', help='Minimal intensity of main peak to report. Default = 1e4.', type=float, default=1e4)
    parser.add_argument('-elements',
                        help='Define the boundaries for elemental composition. Input format [minimal number]-[element]-[maximal number]_[next element]. E.g. "2-C-10_2-N-5_0-H-20". For custom masses, such as between 1 and 2 phenols, add 1-phenol:94.0419-2.',
                        type=str, default=None)
    parser.add_argument('-accuracy',
                        help='Tolerance of relative mass difference between measured and predicted masses. Default = 5e-6.',
                        type=float, default=5e-6)
    parser.add_argument('-time_range',
                        help='Set a custom time range to analyze in the mass spec data. Example: for 3-10 minutes, enter 3-10.',
                        type=str, default='0-1000')
    parser.add_argument('-mass_range',
                        help='Set a custom mass range to analyze in the mass spec data. Example: for m/z 200-600, enter 200-600.',
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
                        help='If monoisotopic is True, the monoisotopic mass peaks will be searched for instead of the calculated highest abundant mass peak. Default: False',
                        action='store_true')
    parser.add_argument('-plot_time_range', help='Time range to use for plotting', default='0-30', type=str)
    parser.add_argument('-plot_mass_range', help='Mass range to use for plotting', default='200-2000', type=str)
    args = parser.parse_args()

    # Print boundary
    sys.stdout.write(f"{''.join(['=' for _ in range(20)])}\n")
    # Check if the input is a directory or file and make a file list
    if os.path.isdir(args.input) and not os.path.isfile(args.input):
        mzxml_files = [os.path.join(args.input, file) for file in os.listdir(args.input) if '.mzxml' in file.lower()]
        sys.stdout.write(f"Detected {len(mzxml_files)} .MZxml files to analyze in {args.input}:\n")
        sys.stdout.write('\t' + '\n\t'.join(mzxml_files) + '\n')
    elif os.path.isfile(args.input) and '.mzxml' in args.input.lower():
        mzxml_files = [args.input]
        sys.stdout.write(f"Analyzing single file: {args.input}.\n")
    else:
        sys.stdout.write(f"Did not detect any MZxml files in {args.input}. Terminating...\n")
        return
    if args.elements is None:
        sys.stdout.write(f"No elements string received. Exiting.\n")
        return

    # Generate variables for the mass and time ranges, formulas with charge, and plot time and mass ranges from the arguments
    mass_range = [float(mass) for mass in args.mass_range.split('-')]
    time_range = [float(time) for time in args.time_range.split('-')]
    formulas_with_charge = generate_formula_with_charge(generate_formulas(args.elements), mass_range, args.monoisotopic)

    # Check if the plot_time_range and plot_mass_range are set to the default values
    # If not, set full_range to True, since full_range is needed to trigger personal plot time and mass ranges
    full_range = args.full_range if args.plot_time_range == '0-30' and args.plot_mass_range == '200-2000' else True

    # Analyze for each file all spectra in parallel. Write output of each file to a txt
    pool = mp.Pool(args.threads)

    nl = '\n\t\t'  # new line for f-strings

    # Check if the output folder is a valid directory
    # If not, save the files in the same directory as the mzxml files
    # If the folder does not exist, create it
    for input_file in mzxml_files:
        sys.stdout.write(f"Started parsing {input_file}\n")
        data = mzxml.MzXML(input_file, use_index=True)
        min_index, max_index = [int(data.time[float(time)]['id']) for time in time_range]
        analyzed_spectra = pool.starmap(analyze_mass_spec, [(data.get_by_index(int(index) - 1), mass_range,
                                                             args.accuracy, formulas_with_charge, args.min_intensity)
                                                            for index in range(min_index, max_index)])
        analyzed_spectra = [spectrum for spectrum in analyzed_spectra if spectrum is not None]
        sys.stdout.write(
            f"\tFound {sum([len(spectrum[1]) for spectrum in analyzed_spectra if spectrum != None])} masses matching the pattern in {input_file}\n")

        # Handle output file path
        if args.output_folder is not None:
            if not os.path.isdir(args.output_folder):
                sys.stdout.write(
                    f"WARNING: {args.output_folder} is not a valid folder path. Saving files in {os.path.dirname(input_file)} instead.\n")
                output_file_path = os.path.join(os.path.dirname(input_file), os.path.basename(input_file))
            else:
                output_file_path = os.path.join(args.output_folder, os.path.basename(input_file))
        else:
            output_file_path = os.path.join(os.path.dirname(input_file), os.path.basename(input_file))

        output_file_path = append_suffix_to_file(os.path.join(os.path.dirname(output_file_path), str(args.output_prefix) + os.path.basename(output_file_path)),
                                     args.overwrite)

        # Write the results to a txt file
        # Check if the relative abundances are not None
        # Otherwise, write the relative abundances of the different formulas

        with open(os.path.splitext(output_file_path)[0] + '_analyzed.txt', 'w') as output_file:
            output_file.write(
                f"Checking for compounds with formulas in range {args.elements}.\nChecking in time range {time_range} and mass range {mass_range}.\n\n")
            output_file.write(
                f"Mass range; {mass_range}\nTime range: {time_range}\n\n")
            output_file.write(
                f"Relative abundances of the different formulas:\n")
            # Check if the relative abundances are not None
            if relative_abundances(generate_plot_list(analyzed_spectra)) is not None:
                for formula, abundance in relative_abundances(generate_plot_list(analyzed_spectra)).items():
                    output_file.write(f"\t{formula}:\t {abundance}\n")
            else:
                output_file.write(f"No matching masses found in {output_file_path}\n")
                continue

            output_file.write(
                f"\nFound {sum([len(spectrum[1]) for spectrum in analyzed_spectra if spectrum != None])} matching masses in {output_file_path}\n")

            for retention_time, spectrum in analyzed_spectra:
                if retention_time is None:
                    continue
                output_file.write(f"Found matching mass at {retention_time} min:\n")
                for peak in spectrum:
                    output_file.write(
                        f"\tExperimental Mass: {peak['experimental_mass']}\tIntensity: {peak['intensity']}\tFormula: {peak['formula']}\tTheoretical mass: {peak['theoretical_mass']}\tParent mass: {peak['parent_mass']}\tCharge state: {peak['charge_state']}\tIsotope peak number: {peak['isotope_peak_number']}" + "\n")

        # Plot the results
        # Check if the full range is set to True
        plot_time_range = [round(float(data.time[float(time)]['retentionTime']), 2) for time in
                           args.plot_time_range.split('-')]
        plot_mass_range = [float(mass) for mass in args.plot_mass_range.split('-')]
        plot_results(analyzed_spectra, os.path.splitext(input_file)[0], plot_time_range, plot_mass_range,
                           args.overwrite, full_range, args.min_intensity)
    pool.close()


if __name__ == '__main__':
    main()

