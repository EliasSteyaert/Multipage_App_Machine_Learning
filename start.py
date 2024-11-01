import pandas as pd 
import re
import numpy as np  

#load the data
topTable = pd.read_csv("topTable.txt", sep="\t", header=0)
targets = pd.read_csv("targets.txt", sep="\t", header=0)

# Replace hyphens with underscores in the 'Sample_ID' column of the sample_info dataframe
targets['Sample_ID'] = targets['Sample_ID'].str.replace('-', '_')
# Use regular expression to make the "Sample_ID" column be the same for both tables
topTable.columns = topTable.columns.str.replace(r'^CPM_([^_]+_[^_]+_[^_]+)_.*$', r'\1', regex=True) 

#print(targets)
#print(topTable.columns)
#print(topTable)

# Set the 'Sample_ID' as index in `table1`
# This ensures that we can access health status for each sample by its Sample ID.
targets.set_index('Sample_ID', inplace=True)

# Step 2: Identify relevant columns in table2 for CPM values
# Using your method to select columns with the required format
cpm_columns = [col for col in topTable.columns if re.match(r'^[A-Z0-9]{2}_[0-9]{4}_[A-Z0-9]{2}$', col)]

# Step 3: Filter table2 to keep only the gene IDs and relevant CPM columns
# We assume 'Ensemble_gene_id' is the first column of table2
topTable_filtered = topTable[['ensembl_gene_id'] + cpm_columns]

# Step 4: Set 'Ensemble_gene_id' as index in table2_filtered
topTable_filtered.set_index('ensembl_gene_id', inplace=True)

# Step 5: Transpose table2_filtered so that each sample ID becomes a row, and each gene becomes a column
table2_T = topTable_filtered.transpose()

# Step 6: Join the health status from table1 with the transposed table2
merged_data = table2_T.join(targets['HLAB27_status'])

# Check the resulting merged data structure
print(merged_data.head())

#check for missing values
#datanul=merged_data.isnull().sum()
#g=[i for i in datanul if i>0]
#
#print('columns with missing values:%d'%len(g))

print("Do we have NA values?")
print(np.any(merged_data.isna()))

print("In case we would have NA values, show these rows")
# Display the na values
print(merged_data[np.any(merged_data.isna(), axis=1)])
print("Empty rows if no NA values")

print("="*79)

print("If we would have NA values, drop the rows containing them")
merged_data.dropna(inplace=True)

#plot a bar chat to display the class distribution
merged_data['HLAB27_status'].value_counts().plot.bar()


#print(topTable_filtered.head())
#print(merged_data)
# At this point, `merged_data` should look like this:
# Rows: each sample (e.g., BP_0081_G2)
# Columns: each gene (e.g., gene_1, gene_2, etc.), followed by a column for 'HLAB27_status'

## Create a mapping from sample_ID to HLAB27_status
#hlab27_mapping = dict(zip(targets['Sample_ID'], targets['HLAB27_status']))
#
## Example function to process the vertical table
#def process_topTable(row):
#    sample_id = row['sample_ID']  # Adjust to the correct column name for sample ID
#    hlab27_status = hlab27_mapping.get(sample_id, None)  # Get the status using the mapping
#    # Use hlab27_status for further processing or features as needed
#    return hlab27_status
#
## Apply the function to each row in the vertical table
#vertical_data['HLAB27_status'] = vertical_data.apply(process_vertical_data, axis=1)
#
## Now vertical_data contains the HLAB27_status linked to each sample without merging the tables
#print(vertical_data.head())

## Create a dictionary for health status
#HLAB27_status_dict = {row['Sample_ID']: row['HLAB27_status'] for _, row in  targets.iterrows()}
#
## Create a dictionary to hold CPM values linked to health statuses
#results = {}
#
## Filter columns in table2 that start with two letters followed by an underscore
#cpm_columns = [col for col in topTable.columns if len(col) > 3 and col[2] == '_']
#
## Loop through each relevant sample ID in table2
#for sample_id in cpm_columns:
#    if sample_id in HLAB27_status_dict:  # Check if the sample ID exists in the health status dict
#        cpm_values = topTable[sample_id].tolist()  # Get the CPM values for the sample ID
#        health_status = HLAB27_status_dict[sample_id]  # Get the health status
#        results[sample_id] = {
#            'HLAB27_status': health_status,
#            'CPM_values': cpm_values
#        }
#
## Display the results
#for sample_id, info in results.items():
#    print(f"Sample ID: {sample_id}, Health Status: {info['HLAB27_status']}, CPM Values: {info['CPM_values']}")


## Select only the relevant columns in table2 (Sample_ID and CPM columns)
#table2_filtered = topTable[['entrezgene'] + cpm_columns]  # Assuming 'entrezgene' is the index in table2
#
## Merge tables on Sample_ID
#merged_data = pd.merge(targets, table2_filtered, left_on='Sample_ID', right_on='entrezgene', how='inner')
#
## Drop the 'entrezgene' column if only needed for the merge
#merged_data.drop(columns=['entrezgene'], inplace=True)
#
## Display the merged data
#print(merged_data)