import os
import sys
import pandas as pd
import re
import numpy as np  # For handling NaN values and calculations

def process_folder(input_folder, output_file=None):
    data = []

    # Loop through subfolders only (skip files in the root directory)
    for subdir, _, files in os.walk(input_folder):
        # Skip processing the root directory itself (only process subdirectories)
        if subdir == input_folder:
            continue  # Skip the root folder and go to subdirectories

        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)

                print(f"Processing file: {file_path}")

                # Extract filename components
                match = re.match(r"(\d{8})_([\w_]+)_analyzed\.txt", file)
                if match:
                    date = match.group(1)
                    name = match.group(2)
                    base_name = f"{date}_{name}"

                    # Check if the filename contains a replicate number at the end first
                    replicate_match = re.search(r"_(\d{1,3})$", name)  # Replicate at the end
                    if replicate_match:
                        replicate = replicate_match.group(1)
                    else:
                        # If no replicate at the end, check for replicate in the middle
                        replicate_match = re.search(r"_(\d{1,3})_", name)  # Replicate in the middle
                        replicate = replicate_match.group(1) if replicate_match else ""

                else:
                    print(f"Skipping file with invalid name format: {file}")
                    continue

                # Process file content
                with open(file_path, 'r') as f:
                    content = f.read()

                # Look for the relevant section
                if "Relative abundances and sum intensities of the different formulas:" in content:
                    section_start = content.split("Relative abundances and sum intensities of the different formulas:", 1)[1].strip()

                    # Check if "No matching masses found" is in the content
                    if "No matching masses found" in section_start:
                        # If "No matching masses found" appears, only add base_name and replicate to the row
                        row = [base_name, replicate]
                        data.append(row)
                        continue

                    # Find the end of the relevant section using the updated regex
                    section_end = re.search(r"Found \d+ matching masses in", section_start)
                    if section_end:
                        section = section_start[:section_end.start()].strip()
                    else:
                        section = section_start

                    # Extract identifiers, relative abundances, and sum intensities using regular expressions
                    identifiers = re.findall(r"Identifier:\s*([^\n]+)", section)
                    abundances = re.findall(r"Relative abundance:\s*(-?\d+\.\d+)", section)
                    sum_intensities = re.findall(r"Sum intensity:\s*(-?\d+\.\d+)", section)

                    # Ensure all lists are of the same length
                    if len(identifiers) == len(abundances) == len(sum_intensities):
                        # Prepare row data, starting with the base name and replicate
                        row = [base_name, replicate]
                        for identifier, abundance, sum_intensity in zip(identifiers, abundances, sum_intensities):
                            row.append(identifier)  # Add identifier to the row
                            row.append(float(abundance))  # Add corresponding relative abundance to the row
                            row.append(float(sum_intensity))  # Add corresponding sum intensity to the row

                        data.append(row)

    if data:
        # Calculate the maximum number of identifier/relative_abundance/sum_intensity triplets in any row
        max_triplets = max((len(row) - 2) // 3 for row in data)  # Exclude "date_name" and "replicate"

        # Create column names: first "date_name", then "replicate", followed by alternating "identifier", "relative_abundance", and "sum_intensity"
        column_names = ["date_name", "replicate"] + [
            f"identifier{i // 3 + 1}" if i % 3 == 0 else f"relative_abundance{i // 3 + 1}" if i % 3 == 1 else f"intensity{i // 3 + 1}"
            for i in range(max_triplets * 3)
        ]

        # Create dataframe
        df = pd.DataFrame(data, columns=column_names)

        # Add a new column 'conversion_rate' initialized to 0
        df['conversion_rate'] = None

        # Loop through the dataframe rows
        for index, row in df.iterrows():
            total_intensity = 0.0
            total_conversion = 0.0

            # Get the identifiers (skip the first two columns: date_name and replicate)
            # skip the last column (conversion_rate)
            identifiers = row[2:-1:3].tolist()
            abundances = row[3::3].tolist()
            intensities = row[4::3].tolist()

            # delete all identifiers that are None and all abundances that are nan
            identifiers = [identifier for identifier in identifiers if identifier is not None]
            abundances = [abundance for abundance in abundances if not pd.isna(abundance)]
            intensities = [intensity for intensity in intensities if not pd.isna(intensity)]

            # if identifiers and abundances are empty, break the loop
            if len(identifiers) == 0 or len(abundances) == 0 or len(intensities) == 0:
               continue

            # Identify the shortest identifier in this row
            shortest_identifier = min(identifiers, key=len)

            # Sum the abundances of identifiers that contain the shortest identifier plus another string
            for i, identifier in enumerate(identifiers):
                total_intensity += intensities[i]
                # Check if the identifier contains the shortest identifier along with some other string
                if shortest_identifier != identifier and shortest_identifier in identifier:
                    total_conversion += abundances[i]

            # Store the summed relative abundance in the 'conversion_rate' column
            df.at[index, 'conversion_rate'] = total_conversion
            df.at[index, 'sum_intensity'] = total_intensity

        # Check replicates, calculate mean/std and store number of replicates used for the mean
        df['mean_conversion_rate'] = None
        df['mean_std_dev'] = None
        df['replicates_used'] = None
        n_rows = len(df)
        i = 0

        while i < n_rows:
            row = df.iloc[i]
            if row['replicate'] == '':
                i += 1
                continue

            # Check for consecutive replicates
            replicate_group = []
            current_replicate = int(row['replicate'])
            while i < n_rows:
                row = df.iloc[i]
                if int(row['replicate']) < current_replicate:
                    break

                if int(row['replicate']) == current_replicate:
                    replicate_group.append(i)
                    current_replicate += 1
                else:
                    break
                i += 1

            # Calculate mean and weighted std for the replicate group
            if replicate_group:
                conversion_rates = df.loc[replicate_group, 'conversion_rate'].dropna()
                sum_intensities = df.loc[replicate_group, 'sum_intensity'].dropna()

                if not conversion_rates.empty and len(conversion_rates) == len(sum_intensities):
                    # Calculate weighted mean
                    weighted_mean = np.average(conversion_rates, weights=sum_intensities)

                    # Calculate weighted std dev
                    if len(conversion_rates) > 1:
                        # Weighted standard deviation formula
                        mean_diff_squared = ((conversion_rates - weighted_mean) ** 2)
                        weighted_variance = np.average(mean_diff_squared, weights=sum_intensities)
                        weighted_std_dev = np.sqrt(weighted_variance)
                    else:
                        weighted_std_dev = 0.0

                    replicates_used = len(conversion_rates)

                    # Store the weighted mean and weighted std dev in the dataframe
                    df.loc[replicate_group, 'mean_conversion_rate'] = weighted_mean
                    df.loc[replicate_group, 'mean_std_dev'] = weighted_std_dev
                    df.loc[replicate_group, 'replicates_used'] = replicates_used

        # Determine the output file name
        if not output_file:
            output_file = os.path.join(input_folder, "output_table.csv")
        elif not output_file.endswith(".csv"):
            output_file += ".csv"

        # Save the dataframe to the output CSV file
        df.to_csv(output_file, index=False)
        print(f"Data successfully saved to {output_file}")
    else:
        print("No valid data found. No CSV file created.")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python process_txt_files.py <folder_path> [output_file]")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else None

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    process_folder(folder_path, output_file)
