import os
import sys
import pandas as pd
import re

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

        # Determine the output file name
        if not output_file:
            output_file = os.path.join(input_folder, "output_table.csv")
        elif not output_file.endswith(".csv"):
            output_file += ".csv"

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
