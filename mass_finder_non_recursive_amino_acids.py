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
import cmocean
from numpy import log10
from math import ceil, floor
from collections import defaultdict
import numpy as np
from platform import system


def analyze_mass_spec(spectrum, mass_range, accuracy, formulas_with_charge, min_intensity):
    # Convert arrays to numpy arrays
    mz_array = np.array(spectrum.get('m/z array', []))
    intensity_array = np.array(spectrum.get('intensity array', []))

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
                                'charge_state': charge
                            })
                            # print(formula, formula_mass, charge, experimental_mass)
                        # Assuming the formulas_with_charge dictionary is organized from low to high mass

    return (float(spectrum.get('retentionTime', 0)), matching_masses) if len(matching_masses) > 0 else None


def find_atomic_mass(element):
    # Dictionary with elemental masses
    atomic_masses = {'H': 1.007825, 'He': 4.002603, 'Li': 7.016005, 'Be': 9.012183,
                     'B': 11.009305, 'C': 12.0, 'N': 14.003074, 'O': 15.994915,
                     'F': 18.998403, 'Ne': 19.992439, 'Na': 22.98977, 'Mg': 23.985045,
                     'Al': 26.981541, 'Si': 27.976928, 'P': 30.973763, 'S': 31.972072,
                     'Cl': 34.968853, 'Ar': 39.962383, 'K': 38.963708, 'Ca': 39.962591,
                     'Sc': 44.955914, 'Ti': 47.947947, 'Cr': 51.94051, 'V': 50.943963,
                     'Fe': 55.934939, 'Mn': 54.938046, 'Ni': 57.935347, 'Co': 58.933198,
                     'Cu': 62.929599, 'Zn': 63.929145, 'Ga': 68.925581, 'Ge': 73.921179,
                     'Se': 79.916521, 'As': 74.921596, 'Kr': 83.911506, 'Br': 78.918336,
                     'Sr': 87.905625, 'Rb': 84.9118, 'Y': 88.905856, 'Zr': 89.904708,
                     'Mo': 97.905405, 'Nb': 92.906378, 'Ru': 101.904348, 'Pd': 105.903475,
                     'Rh': 102.905503, 'Cd': 113.903361, 'Ag': 106.905095, 'Sn': 119.902199,
                     'In': 114.903875, 'Te': 129.906229, 'Sb': 120.903824, 'Xe': 131.904148,
                     'X': 125.904281, 'I': 126.904477, 'Ba': 137.905236, 'Cs': 132.905433,
                     'Ce': 139.905442, 'La': 138.906355, 'Pr': 140.907657, 'Nd': 141.907731,
                     'Sm': 151.919741, 'Eu': 152.921243, 'Gd': 157.924111, 'Dy': 163.929183,
                     'Tb': 158.92535, 'Er': 165.930305, 'Ho': 164.930332, 'Yb': 173.938873,
                     'Tm': 168.934225, 'Hf': 179.946561, 'Lu': 174.940785, 'W': 183.950953,
                     'Ta': 180.948014, 'Os': 191.961487, 'Re': 186.955765, 'Pt': 194.964785,
                     'Ir': 192.962942, 'Hg': 201.970632, 'Au': 196.96656, 'Tl': 204.97441,
                     'Pb': 207.976641, 'Bi': 208.980388, 'Th': 232.038054, 'U': 238.050786,
                     'H+': 1.00728, 'Ac': 42.01056, 'H2O': 18.01056, 'CH2': 14.01565, 'CH1': 13.00703,
                     'Ala': 71.03711, 'Arg': 156.10111, 'Asn': 114.04293, 'Asp': 115.02694,
                     'Cys': 103.00919, 'Glu': 129.04259, 'Gln': 128.05858, 'Gly': 57.02146,
                     'His': 137.05891, 'Ile': 113.08406, 'Leu': 113.08406, 'Lys': 128.09496,
                     'Met': 131.04049, 'Phe': 147.06841, 'Pro': 97.05276, 'Ser': 87.03203,
                     'Thr': 101.04768, 'Trp': 186.07931, 'Tyr': 163.06333, 'Val': 99.06841}
    if element in atomic_masses:
        return atomic_masses[element]


def find_amino_acid_mass(amino_acid):
    # Dictionary with elemental masses
    amino_acid_masses = {'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
                         'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
                         'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
                         'M': 131.04049, 'F': 147.06841, 'P': 97.05276, 'S': 87.03203,
                         'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841}
    if amino_acid in amino_acid_masses:
        return amino_acid_masses[amino_acid]


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
            peptide = list(identifier)
            mass = 0
            for amino_acid in peptide:
                amino_acid_mass = find_amino_acid_mass(amino_acid)
                mass += amino_acid_mass
        # get all atomic masses
        else:
            mass = find_atomic_mass(identifier)
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
            backtrack(updated_formula, current_element + 1, updated_mass)

    backtrack("", 0, 0.0)
    # Sort the formulas dictionary with a lambda function that sorts by the mass
    formulas = {formula: mass for formula, mass in sorted(formulas.items(), key=lambda item: item[1])}

    return formulas


def generate_formula_with_charge(formulas, mass_range, monoisotopic):
    formulas_with_charge = {}
# find maximum and minimum charge states for each formula
    for formula, mass in formulas.items():
        min_charge, max_charge = calculate_charge_range(mass, mass_range)
        isotope_peak_number = int(mass / 1500)
        print(formula, mass, min_charge, max_charge)

        for charge in range(min_charge, max_charge + 1):
            if monoisotopic:
                charge_state_mass = round((mass + (charge * 1.0073)) / charge, 4)
            else:
                # calculate the number of the most abundant 13C isotope peak, empirically determined to change at 1500 Da
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


def plot_results_in_2D(analyzed_spectra, output_file, time_range, mass_range, overwrite, full_range):
    # Plot the analyzed spectra in a single graph
    plot_list = list()
    for spectrum in [spectrum for spectrum in analyzed_spectra if spectrum is not None]:
        time = spectrum[0]
        for peak in spectrum[1:]:
            peaks = [value for value in peak]
            for peak in peaks:
                plot_list.append([peak['experimental_mass'], time, log10(peak['intensity'])])
    plot_list = pd.DataFrame(plot_list, columns=['experimental_mass', 'time', 'intensity'])
    plot_list = plot_list.sort_values(by='intensity', ascending=True, ignore_index=True)

    # Check if plot_list is empty
    if plot_list.empty:
        sys.stdout.write(f"Nothing to plot. The plot list is empty.\n")
        return

    # Define the plot
    zrange = [power for power in range(floor(min(plot_list['intensity'])), ceil(max(plot_list['intensity'])))]
    plt.figure(figsize=(12, 10))
    plt.title(os.path.splitext(os.path.basename(output_file))[0], fontsize=22)
    plot = plt.scatter(plot_list['time'], plot_list['experimental_mass'], c=plot_list['intensity'],
                       marker='.',
                       cmap=cmocean.cm.rain,
                       vmin=zrange[0], vmax=zrange[-1])
    plt.xlabel('Time (min)', fontsize=18)
    plt.ylabel('m/z', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if full_range:
        plt.xlim((min(time_range), max(time_range)))
        plt.ylim((min(mass_range), max(mass_range)))
    else:
        plt.xlim((0.9 * min(plot_list['time']), 1.1 * max(plot_list['time'])))
        plt.ylim((0.9 * min(plot_list['experimental_mass']), 1.1 * max(plot_list['experimental_mass'])))
    plt.grid(which='both', alpha=0.3)
    cbar = plt.colorbar(plot, shrink=0.5)
    cbar.ax.set_ylabel('log(intensity)', rotation=270, labelpad=20, fontsize=18)
    cbar.set_ticks(zrange)
    cbar.ax.tick_params(labelsize=14)

    # Save the plot
    if any([os.path.isfile(output_file + '.svg'), os.path.isfile(output_file + '.png')]) and overwrite:
        suffix = 1
        while any([os.path.isfile(output_file + '_' + str(suffix) + '.svg'),
                   os.path.isfile(output_file + '_' + str(suffix) + '.png')]):
            suffix += 1
        output_file = f"{output_file}_{suffix}"
    plt.savefig(output_file + '.svg', transparent=True, dpi=300)
    plt.savefig(output_file + '.png', transparent=True, dpi=300)


def clean_formula(formula):
    # Remove the parts in the chemical formula that have coefficient 0 and clean
    # the formula
    elements = formula.split('_')
    clean_formula = list()
    chemical_groups = ['C2H4', 'C2H2', 'CH2', 'NH3', 'O']
    for element in elements:
        # Check if the last value is 0, not 10/20.. and not a custom element
        if element[-1] == '0' and not ':' in element:
            if element[-2].isdigit() and not any([alkyl in element[-2 - len(alkyl):-1] for alkyl in chemical_groups]):
                pass
            else:
                continue
        else:
            clean_formula.append(element)
    customs = [element for element in clean_formula if ':' in element]
    adducts = [element for element in clean_formula if '+' in element]
    chemical_groups = [element for element in clean_formula if any([moiety in element for moiety in chemical_groups])]
    remainder = sorted([element for element in clean_formula if
                        not any([element in adducts, element in customs, element in chemical_groups])])
    clean_formula = customs + remainder + chemical_groups + adducts
    return '_'.join(clean_formula)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Check an xml files in a folder for a specific isotope pattern.')
    parser.add_argument('-input', help='Input folder or file')
    parser.add_argument('-threads', help='Number of threads to use', type=int, default=1)
    parser.add_argument('-min_intensity', help='Minimal intensity of main peak to report', type=float, default=1e4)
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
                               args.overwrite, args.full_range)
        else:
            plot_time_range = [round(float(data.time[float(time)]['retentionTime']), 2) for time in args.plot_time_range.split('-')]
            plot_mass_range = [float(mass) for mass in args.plot_mass_range.split('-')]
            plot_results_in_2D(analyzed_spectra, os.path.splitext(file)[0], plot_time_range, plot_mass_range,
                               args.overwrite, args.full_range)
    pool.close()


if __name__ == '__main__':
    main()
