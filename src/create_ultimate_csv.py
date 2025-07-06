import os
import pandas as pd

# --- Define directories ---
FINAL_PROCESSED_DIR = "data/final_processed_no_outliers"
CLEANED_CAPTIONS_DIR = "data/cleaned_captions"
OUTPUT_DIR = "data/output"  # This is where the final combined CSV will be saved

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List all CSV files in the cleaned captions and final_processed_no_outliers folders
cleaned_captions_files = [f for f in os.listdir(CLEANED_CAPTIONS_DIR) if f.endswith('.csv')]
final_processed_files = [f for f in os.listdir(FINAL_PROCESSED_DIR) if f.endswith('.csv')]

# Make sure both folders have the same number of CSV files
if len(cleaned_captions_files) != len(final_processed_files):
    raise ValueError("Mismatch in number of CSV files between cleaned_captions and final_processed_no_outliers.")

# Ensure the files match between both folders (same file names)
cleaned_captions_files.sort()
final_processed_files.sort()

# List to hold final DataFrames
final_combined_data = []

# Process each matching pair of CSVs
for cleaned_file, final_file in zip(cleaned_captions_files, final_processed_files):
    # Load cleaned caption CSV
    cleaned_df = pd.read_csv(os.path.join(CLEANED_CAPTIONS_DIR, cleaned_file))
    
    # Load original metadata CSV
    final_df = pd.read_csv(os.path.join(FINAL_PROCESSED_DIR, final_file))

    # Check that necessary columns are present in both files
    if 'Image Filename' not in cleaned_df.columns or 'Cleaned Caption' not in cleaned_df.columns:
        raise KeyError(f"Required columns missing in {cleaned_file}")
    if 'Image Filename' not in final_df.columns:
        raise KeyError(f"Required columns missing in {final_file}")
    
    # Merge the cleaned caption from the cleaned_df to the final_df using the 'Image Filename'
    combined_df = pd.merge(final_df, cleaned_df[['Image Filename', 'Cleaned Caption']], on='Image Filename', how='left')

    # Append the combined DataFrame to the list
    final_combined_data.append(combined_df)

    # Output progress
    print(f"Processed {cleaned_file} and {final_file}")

# Concatenate all DataFrames from the list into one final DataFrame
ultimate_df = pd.concat(final_combined_data, ignore_index=True)

# Save the combined data to ultimate.csv
ultimate_csv_path = os.path.join(OUTPUT_DIR, 'ultimate.csv')  # Change the path as needed
ultimate_df.to_csv(ultimate_csv_path, index=False)

print(f"\n✅ Ultimate CSV created and saved successfully at: {ultimate_csv_path}")
