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
from platform import system

def find_coefficients_in_mass(element_dictionary, mass, absolute_accuracy):
    # Find the bruto formula of a specific mass within an accuracy range
    stack = []
    formula_list = []
    elements = [element for element in element_dictionary]

    while len(elements) > 1 or stack:
        if stack:
            element, _min, _max, mass_element, truncated_element_dictionary = stack.pop()
        else:
            element = elements[0]
            _min, _max, mass_element = element_dictionary[element]
            truncated_element_dictionary = {element: element_dictionary[element] for element in set(elements[1:])}

        for n_element in range(_min, _max + 1):
            if n_element * mass_element > mass + absolute_accuracy:
                break
            elif abs(n_element * mass_element - mass - sum([element[0] * element[2] for element in truncated_element_dictionary.values()])) < absolute_accuracy:
                formula_list.append(f"{element}{n_element}")
            else:
                stack.append((element, _min, _max, mass_element, truncated_element_dictionary))
                mass -= n_element * mass_element
                elements = [elem for elem in truncated_element_dictionary]
                break

    if formula_list:
        return formula_list
    else:
        if len(elements) == 1:
            _min, _max, mass_element = element_dictionary[elements[0]]
            if (mass - absolute_accuracy) / mass_element < round(mass / mass_element, 0) < (mass + absolute_accuracy) / mass_element:
                n_element = int(round(mass / mass_element, 0))
                if _min <= n_element <= _max:
                    return [f"{elements[0]}{n_element}"]

    return ['None']


def cluster_charge_states(isotope_masses):
    clustered_masses = {}
    for isotope in isotope_masses:
        parent_mass = isotope['mass']
        charge_state = isotope['charge']
        if parent_mass not in clustered_masses:
            clustered_masses[parent_mass] = []
        clustered_masses[parent_mass].append(isotope)
    return clustered_masses

def analyze_mass_spec(spectrum, mass_difference, difference_intensity, min_intensity, element_dictionary, accuracy, mass_range):
    # Parse through a mass spectrum at a specific time and return chlorinated
    # masses
    isotope_masses = list()
    # sys.stdout.write(f"\r\tAnalyzing spectrum {spectrum['retentionTime']}")

    for index, mass in enumerate(spectrum['m/z array']):
        if mass < mass_range[0] or mass > mass_range[1]:
            continue
        for delta_index, isotope_mass in enumerate(spectrum['m/z array'][index:]):
            if isotope_mass - mass > mass_difference*1.1:
                break
            if abs(mass + mass_difference-isotope_mass) < accuracy*mass and \
                difference_intensity[0] < abs(spectrum['intensity array'][index+delta_index]/spectrum['intensity array'][index]) < difference_intensity[1]\
                    and spectrum['intensity array'][index] > min_intensity:
                if element_dictionary is not None:
                    formulas = find_coefficients_in_mass(element_dictionary, mass, mass * accuracy)
                    if formulas != []:
                        isotope_masses.append({'index': index,
                                               'intensity': round(spectrum['intensity array'][index], 0),
                                               'isotope index': index + delta_index,
                                               'isotope_intensity': spectrum['intensity array'][index + delta_index],
                                               'mass': round(mass, 4),
                                               'charge': spectrum['charge array'][index],
                                               'formulas': formulas})
                else:
                    isotope_masses.append({'index': index,
                                           'intensity': round(spectrum['intensity array'][index], 0),
                                           'isotope index': index + delta_index,
                                           'isotope_intensity': spectrum['intensity array'][index + delta_index],
                                           'mass': round(mass, 4),
                                           'charge': spectrum['charge array'][index]})
                return (float(spectrum['retentionTime']), cluster_charge_states(isotope_masses)) if len(
                    isotope_masses) > 0 else None
    return (float(spectrum['retentionTime']), cluster_charge_states(isotope_masses)) if len(isotope_masses) > 0 else None


def find_atomic_mass(element):
    # Dictionary with elemental masses
    element = element.split('-')[1]
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
                     'CH2': 14.0157, 'C2H4': 28.0313, 'C2H2': 26.0157, 'NH3': 17.0265,
                     'deca': 171.1385, 'undeca': 185.1542, 'dodeca': 199.1698, 'trideca': 213.1855,
                     'tetradeca': 227.2011, 'pentadeca': 241.2168, 'hexadeca': 255.2324,
                     'heptadeca': 269.2481, 'octadeca': 283.2637, 'hexadece': 254.2246,
                     'octadece': 281.2481, '3OHdeca': 187.1334, '3OH5dodece': 213.1491,
                     '3OHdodeca': 215.1647, 'methylenehexadeca': 267.2324,
                     'methyleneoctadeca': 295.2637, 'propyl': 73.0290, 'butyl': 87.0446,
                     'pentyl': 101.0603, 'hexyl': 115.0759, 'heptyl': 129.0916,
                     'octyl': 143.1072, 'nonyl': 157.1229, 'ethyl': 59.0133, 'H+': 1.0073}
    if element in atomic_masses:
        return atomic_masses[element]
    else:
        # If a custom mass, such as 'phenol:94.0419' is provided, get this one.
        return float(element.split(':')[1])


def construct_element_dictionary(element_string):
    # Construct the element dictionary
    if element_string is None:
        return None
    elements = element_string.split('_')
    for element in elements:
        assert len(element.split('-')) == 3
    return {element.split('-')[1]: [int(element.split('-')[0]), int(element.split('-')[2]), find_atomic_mass(element)] for element in elements}


def append_suffix_to_file(file, overwrite):
    # Check if files already exist and if so, append a suffix to the file name
    file = os.path.splitext(file)[0]
    if overwrite:
        return file
    if any([os.path.isfile(file+extension) for extension in ['.svg', '.png', '_analyzed.txt']]):
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
                plot_list.append([peak['mass'], time, log10(peak['intensity'])])
    plot_list = pd.DataFrame(plot_list, columns=['mass', 'time', 'intensity'])
    plot_list = plot_list.sort_values(by='intensity', ascending=True, ignore_index=True)

    # Define the plot
    zrange = [power for power in range(floor(min(plot_list['intensity'])), ceil(max(plot_list['intensity'])))]
    plt.figure(figsize=(12, 10))
    plt.title(os.path.splitext(os.path.basename(output_file))[0], fontsize=22)
    plot = plt.scatter(plot_list['time'], plot_list['mass'], c=plot_list['intensity'],
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
        plt.xlim((0.9*min(plot_list['time']), 1.1*max(plot_list['time'])))
        plt.ylim((0.9*min(plot_list['mass']), 1.1*max(plot_list['mass'])))
    plt.grid(which='both', alpha=0.3)
    cbar = plt.colorbar(plot, shrink=0.5)
    cbar.ax.set_ylabel('log(intensity)', rotation=270, labelpad=20, fontsize=18)
    cbar.set_ticks(zrange)
    cbar.ax.tick_params(labelsize=14)

    # Save the plot
    if any([os.path.isfile(output_file+'.svg'), os.path.isfile(output_file+'.png')]) and overwrite:
        suffix = 1
        while any([os.path.isfile(output_file+'_'+str(suffix)+'.svg'), os.path.isfile(output_file+'_'+str(suffix)+'.png')]):
            suffix += 1
        output_file = f"{output_file}_{suffix}"
    plt.savefig(output_file+'.svg', transparent=True, dpi=300)
    plt.savefig(output_file+'.png', transparent=True, dpi=300)

def clean_formula(formula):
    # Remove the parts in the chemical formula that have coefficient 0 and clean
    # the formula
    elements = formula.split('_')
    clean_formula = list()
    chemical_groups = ['C2H4','C2H2','CH2','NH3', 'O']
    for element in elements:
        # Check if the last value is 0, not 10/20.. and not a custom element
        if element[-1] == '0' and not ':' in element:
            if element[-2].isdigit() and not any([alkyl in element[-2-len(alkyl):-1] for alkyl in chemical_groups]):
                pass
            else:
                continue
        else:
            clean_formula.append(element)
    customs = [element for element in clean_formula if ':' in element]
    adducts = [element for element in clean_formula if '+' in element]
    chemical_groups = [element for element in clean_formula if any([moiety in element for moiety in chemical_groups])]
    remainder = sorted([element for element in clean_formula if not any([element in adducts, element in customs, element in chemical_groups])])
    clean_formula = customs + remainder + chemical_groups + adducts
    return '_'.join(clean_formula)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Check an xml files in a folder for a specific isotope pattern.')
    parser.add_argument('-input', help='Input folder or file')
    parser.add_argument('-mass_difference', help='Mass difference to search. Default is chlorine', type=float, default=1.99704989)
    parser.add_argument('-threads', help='Number of threads to use', type=int, default=1)
    parser.add_argument('-min_intensity', help='Minimal intensity of main peak to report', type=float, default=1e4)
    parser.add_argument('-elements', help='Define the boundaries for elemental composition. Input format [minimal number]-[element]-[maximal number]_[next element]. E.g. "2-C-10_2-N-5_0-H-20]". For custom masses, such as between 1 and 2 phenols, add 1-phenol:94.0419-2.', type=str, default=None)
    parser.add_argument('-accuracy', help='Tolerance of relative mass difference between measured and predicted masses. Default = 5e-6.', type=float, default=5e-6)
    parser.add_argument('-time_range', help='Set a custom time range to analyze in the mass spec data. Example: for 3-10 minutes, enter 3-10.', type=str, default='0-1000')
    parser.add_argument('-mass_range', help='Set a custom mass range to analyze in the mass spec data. Example: for m/z 200-600 , enter 200-600.', type=str, default='0-1000')
    parser.add_argument('-output_folder', help='Specify a specific output folder. If not specified, the output will be in the same folder as the mzxml files.', type=str)
    parser.add_argument('-output_prefix', help='Prefix to the output files', default=None, type=str)
    parser.add_argument('-overwrite', help='If overwrite is True, the data saved from previous runs will be overwritten.',  action='store_true')
    parser.add_argument('-full_range', help='If full_range is True, the output plot will span the entire time and mass range. Useful for comparing samples, but less ideal to check a single file. Default: False',  action='store_true')
    parser.add_argument('-plot_time_range',help='Time range to use for plotting', default='0-30', type=str)
    parser.add_argument('-plot_mass_range',help='Mass range to use for plotting', default='0-1000', type=str)
    parser.add_argument('-intensity_diff', help='Min and max intensity ratios of isotope peaks to be detected. Default is for chlorination: "0.25-1".', default = '0.25-1', type=str)
    
    args = parser.parse_args()

    # Print boundary
    sys.stdout.write(f"{''.join(['=' for _ in range(20)])}\n")
    # Check if the input is a directory or file and make a file list
    if os.path.isdir(args.input) and not os.path.isfile(args.input):
        files = [os.path.join(args.input, file) for file in os.listdir(args.input) if '.mzxml' in file.lower()]
        sys.stdout.write(f"Detected {len(files)} .MZxml files to analyze in {args.input}:\n")
        sys.stdout.write('\t'+'\n\t'.join(files)+'\n')
    elif os.path.isfile(args.input) and '.mzxml' in args.input.lower():
        files = [args.input]
        sys.stdout.write(f"Analyzing single file: {args.input}.\n")
    else:
        sys.stdout.write(f"Did not detect any MZxml files in {args.input}. Terminating...\n")
        return
    if args.elements is None:
        sys.stdout.write(f"No elements string received. Exiting.\n")
        return
    if args.full_range and args.plot_mass_range == '0-1000':
        operating_system = system()
        if operating_system == 'Windows':
            reply = ''
        else:
            sys.stdout.write(f"Running on Linux. Using standard mass_range: 0-1000.\n")
            reply = 'n'
        while reply.lower() not in ['y', 'n']:
            reply = input('No mass range specified? The original range is not saved in the mzXML files. Do you want to specify your own instead of using the default 0-1000? [y/n]:\n')
        if reply.lower == 'y':
            args.plot_mass_range = input("Input your desired mass range:\n")
    difference_intensities = [float(ratio) for ratio in args.intensity_diff.split('-')]

    # Analyze for each file all spectra in parallel. Write output of each file to a txt
    pool = mp.Pool(args.threads)
    element_dictionary = construct_element_dictionary(args.elements)
    mass_range = [float(mass) for mass in args.mass_range.split('-')]
    time_range = [float(time) for time in args.time_range.split('-')]
    nl = '\n\t\t'  # new line for f-strings

    for file in files:
        sys.stdout.write(f"Started parsing {file}\n")
        data = mzxml.MzXML(file, use_index=True, charge_array=True)
        min_index, max_index = [int(data.time[float(time)]['id']) for time in time_range]
        analyzed_spectra = pool.starmap(analyze_mass_spec, [(data.get_by_index(int(index)-1), args.mass_difference, difference_intensities, args.min_intensity, element_dictionary, args.accuracy, mass_range)
                                                            for index in range(min_index, max_index)])
        analyzed_spectra = [spectrum for spectrum in analyzed_spectra if spectrum is not None]
        sys.stdout.write(f"\tFound {sum([len(spectrum[1]) for spectrum in analyzed_spectra if spectrum != None])} masses matching the pattern in {file}\n")

        if args.output_folder is not None:
            if not os.path.isdir(args.output_folder):
                sys.stdout.write(f"WARNING: {args.output_folder} is not a valid folder path. Saving files in {os.path.dirname(file)} instead.\n")
            else:
                file = os.path.join(args.output_folder, os.path.basename(file))
        file = append_suffix_to_file(os.path.join(os.path.dirname(file), args.output_prefix + os.path.basename(file)), args.overwrite)
        with open(os.path.splitext(file)[0]+'_analyzed.txt', 'w') as output_file:
            output_file.write(f"Checking for compounds with formulas in range {args.elements}.\nChecking in time range {time_range} and mass range {mass_range}.\n\n")
            output_file.write(f"Mass range; {mass_range}\nTime range: {time_range}\nMass difference: {args.mass_difference}\nIntensity ratios: {args.intensity_diff}\n\n")
            output_file.write(f"Found {sum([len(spectrum[1]) for spectrum in analyzed_spectra if spectrum != None])} masses matching the pattern in {file}\n")

            for retention_time, spectrum in analyzed_spectra:
                if retention_time is None:
                    continue
                output_file.write(f"Found matching pattern at {retention_time} min:\n")
                for peak in spectrum:
                    output_file.write(f"\tMass: {peak['mass']}\tIntensity: {peak['intensity']}{nl}{nl.join([clean_formula(formula) for formula in peak['formulas']])}\n")
        if args.full_range:
            plot_time_range = [int(data.time[float(time)]['retentionTime']) for time in [0,1000]]
            plot_mass_range = [float(mass) for mass in args.plot_mass_range.split('-')]
            plot_results_in_2D(analyzed_spectra, os.path.splitext(file)[0], plot_time_range, plot_mass_range, args.overwrite, args.full_range)
        else:
            plot_time_range = [int(data.time[float(time)]['retentionTime']) for time in args.plot_time_range.split('-')]
            plot_mass_range = [float(mass) for mass in args.plot_mass_range.split('-')]
            plot_results_in_2D(analyzed_spectra, os.path.splitext(file)[0], plot_time_range, plot_mass_range, args.overwrite, args.full_range)
    pool.close()


if __name__ == '__main__':
    main()
