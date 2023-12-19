import pandas as pd
import glob

# List all JSON files
json_files = glob.glob("/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskWomen_download/AskWomen_per_year_comments_14M_adjusted_preprocessed_2_counted/processed_preprocessed_sampled_comments_*.json")

# Initialize an empty list to hold DataFrames
dfs = []

# Read each JSON file into a DataFrame, drop 'body' column, and append to list
for file in json_files:
    print(f"Reading {file}")
    df = pd.read_json(file, lines=True)
    df.drop(columns=['body'], inplace=True)  # Drop the 'body' column
    dfs.append(df)
    # Free up memory
    del df

print('Finished reading the files.')

# Concatenate all DataFrames
final_df = pd.concat(dfs, ignore_index=True)

print("Concatenated the DataFrames.")

# Free up memory
del dfs

print('Deleted the list of DataFrames. Starting to write the final DataFrame.')

# Save the concatenated DataFrame to a new CSV file with tab separator
final_df.to_csv("/pfs/work7/workspace/scratch/ma_sguliyev-Reddit_dumps/AskWomen_download/AskWomen_per_year_comments_14M_adjusted_preprocessed_2_counted/AskWomen_combined_comments.csv", index=False, sep='\t')

print('Finished writing the final DataFrame.')
