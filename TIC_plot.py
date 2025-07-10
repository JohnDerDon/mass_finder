import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pyteomics import mzxml

# Define color palette
color_palette = {
    "Pink1": "#FF00FF", "Pink2": "#D20073", "Pink3": "#800080", "Pink4": "#FF99CC", "Pink5": "#FFB3FF", "Pink6": "#D274C5",
    "Gray1": "#000000", "Gray2": "#4D4D4D", "Gray3": "#AEAEAE", "Gray4": "#D1D1D1", "Gray5": "#6D5B5B", "Gray6": "#4B4B65",
    "Yellow1": "#FFC700", "Yellow2": "#FF9900", "Yellow3": "#CC9900", "Yellow4": "#FFCC66", "Yellow5": "#FFEC00", "Yellow6": "#FFFF66",
    "Green1": "#009900", "Green2": "#005500", "Green3": "#2ECC40", "Green4": "#669900", "Green5": "#339966", "Green6": "#8ED973",
    "Blue1": "#1C0ED8", "Blue2": "#1877CE", "Blue3": "#000099", "Blue4": "#00FFFF", "Blue5": "#89C1FF", "Blue6": "#5807F9",
    "Red1": "#C00000", "Red2": "#FF2D2D", "Red3": "#D86E6E", "Red4": "#820000", "Red5": "#F9B08F", "Red6": "#EE4F08"
}

def extract_tic_from_spectrum(spectrum, time_range, mass_range):
    if spectrum.get('msLevel', 0) != 1:
        return None

    mz_array = np.array(spectrum.get('m/z array', []))
    intensity_array = np.array(spectrum.get('intensity array', []))
    rt = float(spectrum.get('retentionTime', 0))

    if mz_array.size == 0 or intensity_array.size == 0:
        return None

    if not (time_range[0] <= rt <= time_range[1]):
        return None

    mask = (mz_array >= mass_range[0]) & (mz_array <= mass_range[1])
    total_intensity = float(np.sum(intensity_array[mask]))
    return (rt, total_intensity)

def generate_unique_filename(base_path):
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    counter = 1
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def plot_TIC(tic_data, time_range, plot_color, output_path, custom_plot_size, title_text, max_intensity=None):
    times, intensities = zip(*sorted(tic_data))

    plt.figure(figsize=(10, 4))
    plt.plot(times, intensities, color=color_palette.get(plot_color, "#000000"), linewidth=1.5)

    max_intensity_actual = max(intensities)
    max_time = times[intensities.index(max_intensity_actual)]

    plt.title(title_text, fontsize=22)
    plt.xlabel('Time (min)', fontsize=18)
    plt.ylabel('Total Intensity', fontsize=18)
    plt.tick_params(axis='both', labelsize=18)

    if custom_plot_size:
        plt.xlim(time_range)
    else:
        plt.xlim(0.9 * min(times), 1.1 * max(times))

    if max_intensity is not None:
        plt.ylim(0, max_intensity)
    else:
        plt.ylim(0, max_intensity_actual * 1.1)

    plt.annotate(f'{max_time:.2f}', xy=(max_time, max_intensity_actual),
                 ha='center', va='bottom', fontsize=18,
                 color=color_palette.get(plot_color, "#000000"))

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot Total Ion Chromatogram (TIC) from mzXML.")
    parser.add_argument('-input', help='Input mzXML file', required=True)
    parser.add_argument('-plot_color', help='Define the plot color. Example: Gray1.', default='Gray1', type=str)
    parser.add_argument('-plot_time_range', help='Time range to display (e.g. 1.0,10.0)', default=None)
    parser.add_argument('-plot_mass_range', help='Mass range to display (e.g. 300.0,1500.0)', default=None)
    parser.add_argument('-overwrite', action='store_true', help='Overwrite existing output file')
    parser.add_argument('-max_intensity', type=float, default=None,
                        help='Set maximum intensity for Y axis (for comparable plots).')
    args = parser.parse_args()

    # Parse ranges
    time_range = (0, float('inf')) if args.plot_time_range is None else tuple(map(float, args.plot_time_range.split(',')))
    mass_range = (0, float('inf')) if args.plot_mass_range is None else tuple(map(float, args.plot_mass_range.split(',')))
    custom_plot_size = args.plot_time_range is not None

    # Generate output filename in same folder as input, with _TIC.svg suffix
    input_dir = os.path.dirname(os.path.abspath(args.input))
    input_base = os.path.splitext(os.path.basename(args.input))[0]
    full_output_path = os.path.join(input_dir, f"{input_base}_TIC.svg")

    if os.path.exists(full_output_path) and not args.overwrite:
        full_output_path = generate_unique_filename(full_output_path)

    title_text = input_base

    spectra = list(mzxml.read(args.input))

    with Pool() as pool:
        tic_data = pool.starmap(extract_tic_from_spectrum, [(s, time_range, mass_range) for s in spectra])

    tic_data = [entry for entry in tic_data if entry is not None]

    if len(tic_data) == 0:
        print("No valid MS1 spectra found in the specified ranges.")
        return

    plot_TIC(tic_data, time_range, args.plot_color, full_output_path, custom_plot_size, title_text, args.max_intensity)
    print(f"TIC plot saved to {full_output_path}")

if __name__ == "__main__":
    main()
