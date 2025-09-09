import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import os
import re

# ------------------ PARAMETERS (tune these) ------------------
MIN_POINTS = 5
PROMINENCE_REL = 0.01
SMOOTH_WINDOW = 5
SMOOTH_POLYORDER = 2
BASELINE_PERCENTILE = 10
NOISE_MULTIPLIER = 3.0
FALLBACK_FRAC = 0.10
MAX_EXTENSION_POINTS = 200
INPUT_FOLDER = r'C:\Users\jeckert\Documents\mass_pattern_finder\function_Daniel\Substrates\mzXML\peak_picking'
OUTPUT_FOLDER = r'C:\Users\jeckert\Documents\mass_pattern_finder\function_Daniel\Substrates\mzXML\peak_picking'
SUMMARY_FILE = "all_peaks_summary.tsv"

# ------------------ HELPERS ------------------

def estimate_baseline_and_noise(intensity, window=51, percentile=BASELINE_PERCENTILE):
    n = len(intensity)
    half = window // 2
    baseline = np.zeros(n)
    mad = np.zeros(n)
    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        seg = intensity[l:r]
        baseline[i] = np.percentile(seg, percentile)
        mad[i] = np.median(np.abs(seg - np.median(seg)))
    return baseline, mad

def smooth(intensity):
    if len(intensity) >= SMOOTH_WINDOW:
        return savgol_filter(intensity, SMOOTH_WINDOW, SMOOTH_POLYORDER)
    else:
        return intensity.copy()

# ------------------ CORE PEAK PICKING ------------------

def find_peaks_with_boundaries(time, intensity):
    time = np.asarray(time)
    intensity = np.asarray(intensity)

    smooth_int = smooth(intensity)
    baseline, mad = estimate_baseline_and_noise(smooth_int, window=min(101, len(smooth_int)))
    noise_floor = baseline + NOISE_MULTIPLIER * mad

    abs_prom = max(smooth_int.max() * PROMINENCE_REL, 1e-9)
    peaks_idx, _ = find_peaks(smooth_int, prominence=abs_prom)

    results = []
    n = len(smooth_int)
    local_minima = np.where((smooth_int <= np.roll(smooth_int, 1)) & (smooth_int <= np.roll(smooth_int, -1)))[0]

    for p in peaks_idx:
        peak_max = smooth_int[p]

        left = None
        left_min_candidates = local_minima[local_minima < p]
        if left_min_candidates.size > 0:
            left_min = left_min_candidates[-1]
            if smooth_int[left_min] <= peak_max * 0.95:
                left = left_min

        if left is None:
            i = p
            steps = 0
            while i > 0 and steps < MAX_EXTENSION_POINTS:
                i -= 1
                steps += 1
                if smooth_int[i] < noise_floor[i] or smooth_int[i] < peak_max * FALLBACK_FRAC:
                    left = i
                    break
            if left is None:
                left = max(0, p - min(MAX_EXTENSION_POINTS, p))

        right = None
        right_min_candidates = local_minima[local_minima > p]
        if right_min_candidates.size > 0:
            right_min = right_min_candidates[0]
            if smooth_int[right_min] <= peak_max * 0.95:
                right = right_min

        if right is None:
            i = p
            steps = 0
            while i < n - 1 and steps < MAX_EXTENSION_POINTS:
                i += 1
                steps += 1
                if smooth_int[i] < noise_floor[i] or smooth_int[i] < peak_max * FALLBACK_FRAC:
                    right = i
                    break
            if right is None:
                right = min(n - 1, p + min(MAX_EXTENSION_POINTS, n - 1 - p))

        left = int(left)
        right = int(right)
        if right - left + 1 < MIN_POINTS:
            continue

        auc = np.trapz(intensity[left:right + 1], time[left:right + 1])

        results.append({
            'peak_idx': int(p),
            'peak_time': float(time[p]),
            'peak_intensity': float(intensity[p]),
            'left_time': float(time[left]),
            'right_time': float(time[right]),
            'auc': float(auc)
        })

    return results

# ------------------ CUSTOM TXT PARSER ------------------

def parse_txt_file(filepath):
    records = []
    with open(filepath, 'r') as f:
        current_time = None
        for line in f:
            line = line.strip()
            time_match = re.match(r'Found matching mass at ([0-9.]+) min:', line)
            if time_match:
                current_time = float(time_match.group(1))
                continue

            if current_time is not None and line.startswith("Experimental Mass:"):
                fields = line.split('\t')
                mass = intensity = identifier = None
                for field in fields:
                    field = field.strip()
                    if field.startswith("Experimental Mass:"):
                        mass = float(field.split(':')[1].strip())
                    elif field.startswith("Intensity:"):
                        intensity = float(field.split(':')[1].strip())
                    elif field.startswith("Identifier:"):
                        identifier = field.split(':')[1].strip()
                if mass is not None and intensity is not None and identifier is not None:
                    records.append({'Time': current_time,
                                    'Identifier': identifier,
                                    'Intensity': intensity})
    df = pd.DataFrame(records)
    df = df.groupby(['Time', 'Identifier'], as_index=False)['Intensity'].sum()
    return df

# ------------------ FILE PROCESSING ------------------

def process_file(input_file):
    df = parse_txt_file(input_file)
    all_results = []

    # First, find peaks for all identifiers
    for identifier, group in df.groupby('Identifier'):
        group_sorted = group.sort_values('Time')
        time = group_sorted['Time'].values
        intensity = group_sorted['Intensity'].values

        peaks = find_peaks_with_boundaries(time, intensity)
        for p in peaks:
            p['identifier'] = identifier
            all_results.append(p)

    # Then, compute total AUC across all peaks in the file
    total_auc_file = sum([p['auc'] for p in all_results])
    for p in all_results:
        p['rel_area_pct'] = 100.0 * p['auc'] / total_auc_file if total_auc_file > 0 else 0.0

    return all_results

def write_results(results, output_file):
    with open(output_file, 'w') as fh:
        fh.write('Identifier\tPeakTime\tPeakIntensity\tLeftTime\tRightTime\tAUC\tRelArea(%)\n')
        for r in results:
            fh.write(f"{r['identifier']}\t{r['peak_time']:.4f}\t{r['peak_intensity']:.1f}\t"
                     f"{r['left_time']:.4f}\t{r['right_time']:.4f}\t{r['auc']:.1f}\t{r['rel_area_pct']:.2f}\n")

# ------------------ MAIN FOLDER PROCESSING ------------------

if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    summary_rows = []

    for file in os.listdir(INPUT_FOLDER):
        if not file.endswith('.txt'):
            continue
        filepath = os.path.join(INPUT_FOLDER, file)
        results = process_file(filepath)
        if not results:
            continue

        output_file = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(file)[0]}_peaks.txt")
        write_results(results, output_file)

        for r in results:
            summary_rows.append({
                'File': file,
                'Identifier': r['identifier'],
                'PeakTime': r['peak_time'],
                'PeakIntensity': r['peak_intensity'],
                'LeftTime': r['left_time'],
                'RightTime': r['right_time'],
                'AUC': r['auc'],
                'RelArea(%)': r['rel_area_pct']
            })
        print(f"Processed {file} → {output_file}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(OUTPUT_FOLDER, SUMMARY_FILE), sep='\t', index=False)
        print(f"Master summary written to {os.path.join(OUTPUT_FOLDER, SUMMARY_FILE)}")
