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
                if "Relative abundances of the different formulas:" in content:
                    section_start = content.split("Relative abundances of the different formulas:", 1)[1].strip()

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

                    # Extract formula and relative abundance pairs using a modified regular expression
                    entries = re.findall(r"([^\s:]+(?:[^\n:]*))\s*:\s*(-?\d+\.\d+)", section)
                    if not entries:
                        continue

                    # Prepare row data, starting with the base name and replicate
                    row = [base_name, replicate]
                    for identifier, abundance in entries:
                        row.append(identifier)  # Add identifier to the row
                        row.append(float(abundance))  # Add corresponding abundance to the row

                    data.append(row)

    if data:
        # Calculate the maximum number of identifier/relative_abundance pairs in any row
        max_pairs = max((len(row) - 2) // 2 for row in data)  # Exclude "date_name" and "replicate"

        # Create column names: first "date_name", then "replicate", followed by alternating "identifier" and "relative_abundance"
        column_names = ["date_name", "replicate"] + [
            f"identifier{i // 2 + 1}" if i % 2 == 0 else f"relative_abundance{i // 2 + 1}"
            for i in range(max_pairs * 2)
        ]

        # Create dataframe
        df = pd.DataFrame(data, columns=column_names)

        # Add a new column 'conversion_rate' initialized to 0
        df['conversion_rate'] = None

        # Loop through the dataframe rows
        for index, row in df.iterrows():
            total_conversion = 0.0

            # Get the identifiers (skip the first two columns: date_name and replicate)
            # skip the last column (conversion_rate)
            identifiers = row[2:-1:2].tolist()
            abundances = row[3::2].tolist()

            # delete all identifiers that are None and all abundances that are nan
            identifiers = [identifier for identifier in identifiers if identifier is not None]
            abundances = [abundance for abundance in abundances if not pd.isna(abundance)]

            # if identifiers and abundances are empty, break the loop
            if len(identifiers) == 0 or len(abundances) == 0:
               continue

            # Identify the shortest identifier in this row
            shortest_identifier = min(identifiers, key=len)

            # Sum the abundances of identifiers that contain the shortest identifier plus another string
            for i, identifier in enumerate(identifiers):
                # Check if the identifier contains the shortest identifier along with some other string
                if shortest_identifier != identifier and shortest_identifier in identifier:
                    total_conversion += abundances[i]

            # Store the summed relative abundance in the 'conversion_rate' column
            df.at[index, 'conversion_rate'] = total_conversion

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

            # Calculate mean and std for the replicate group
            if replicate_group:
                conversion_rates = df.loc[replicate_group, 'conversion_rate'].dropna()

                if not conversion_rates.empty:
                    mean_conversion = conversion_rates.mean()
                    replicates_used = len(conversion_rates)

                    # Calculate std dev, if only one value, set std_dev to 0.0
                    if len(conversion_rates) > 1:
                        std_dev = conversion_rates.std()
                    else:
                        std_dev = 0.0

                    df.loc[replicate_group, 'mean_conversion_rate'] = mean_conversion
                    df.loc[replicate_group, 'mean_std_dev'] = std_dev
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
