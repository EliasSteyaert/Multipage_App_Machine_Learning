import streamlit as st
import pandas as pd
import re

# Step 1: File Uploads and Initial Settings
uploaded_files = st.file_uploader("Upload your files (one or two)", accept_multiple_files=True, type=["txt", "csv"])

if uploaded_files:
    # Initialize file settings and placeholders
    file_settings = {}
    file_names = []  # List to store original filenames
    for idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name  # Use the original filename
        file_names.append(file_name)  # Store the filename
        with st.expander(f"Settings for {file_name}", expanded=True):
            # Settings for headers, separator, decimal indicator, etc.
            has_header = st.checkbox(f"Header present in {file_name}", key=f"header_{idx}")
            separator = st.radio(f"Separator for {file_name}:", options=["Tab", "Comma", "Semicolon"], key=f"separator_{idx}", horizontal=True)
            sep_dict = {"Tab": "\t", "Comma": ",", "Semicolon": ";"}
            sep = sep_dict[separator]
            decimal = st.radio(f"Decimal indicator for {file_name}:", options=[",", "."], key=f"decimal_{idx}", horizontal=True)

            # Load file with user-defined settings
            try:
                df = pd.read_csv(uploaded_file, sep=sep, header=0 if has_header else None, decimal=decimal)
                file_settings[file_name] = {"dataframe": df}
                st.write(f"Preview of {file_name}")
                st.write(df.head())
            except Exception as e:
                st.error(f"Error loading {file_name}: {e}")
                
	# Step 2: Data processing when the user uploaded only one file
    if len(uploaded_files) == 1:
        with st.expander("Single File Processing", expanded=True):
            single_file_name = file_names[0]
            data = file_settings[single_file_name]["dataframe"]
            
			# Column inputs for illness status and sample ID's
            illness_status_col = st.text_input("Enter the column name that indicates patient illness status:")
            sample_id_col = st.text_input("Enter the column name that holds sample IDs:")

    # Step 3: Data Processing when the user uploaded two files
    if len(uploaded_files) == 2:
        with st.expander("File Type Selection", expanded=True):
            data_type_topTables = st.radio(f"Which file contains gene expression data?", file_names)
            data_type_targets = file_names[1] if data_type_topTables == file_names[0] else file_names[0]
            
            # Reference the dataframes based on user selection
            topTables = file_settings[data_type_topTables]["dataframe"]
            targets = file_settings[data_type_targets]["dataframe"]
            
            # Column inputs for illness status, sample IDs and gene_names
            gene_name = st.text_input("Enter the column name in the gene expression table that holds the gene names or ID's:")
            illness_status_col = st.text_input("Enter the column name in sample data that indicates patient illness status:")
            sample_id_col = st.text_input("Enter the column name in sample data that holds sample IDs:")

            # Step 3: Data Processing - Find matching headers and rename them
            if illness_status_col and sample_id_col:
                # Replace hyphens with underscores in the sample ID column of the targets DataFrame
                targets[sample_id_col] = targets[sample_id_col].str.replace('-', '_')

                # Extract sample IDs from targets DataFrame
                sample_ids = targets[sample_id_col].tolist()  # Make sure to use the correct column

                # Initialize a dictionary to hold old header names and their corresponding sample IDs
                rename_dict = {}
                new_column_names = []  # List to store new column names

                # Loop through sample IDs and find corresponding headers in topTables
                for sample_id in sample_ids:
                    # Create a regex pattern that allows for any characters before and after the sample ID
                    pattern = re.compile(f".*{re.escape(sample_id)}.*")

                    # Check each header in the topTables DataFrame
                    for header in topTables.columns:
                        if pattern.match(str(header)):
                            # Map old header name to the new sample ID
                            rename_dict[header] = sample_id
                            new_column_names.append(sample_id)  # Store the new name

                # Rename the columns in topTables based on the rename_dict
                if rename_dict:  # Check if there's any column to rename
                    topTables.rename(columns=rename_dict, inplace=True)

                # Step 4: Create a new DataFrame with only the relevant columns
                cpm_columns = topTables[new_column_names].copy()  # Copy only the renamed columns
                relevant_columns = pd.concat([topTables[[gene_name]], cpm_columns], axis=1)  # Include the gene name column
                # Reset the index to remove the default index
                relevant_columns.reset_index(drop=True, inplace=True)
                #st.write("Relevant Columns After Adjustments:", relevant_columns.head(), use_container_width=True)
				#ENDING ON TRYING TO DELETE THE GHOST COLUMN AT THE RELEVENT COLUMNS PART


				#converting to one "data" dataframe
                # Step 1: Set the 'Sample_ID' as index in `table1`
                targets.set_index(sample_id_col, inplace=True)
                # Step 2: Filter topTables to keep only the gene IDs and relevant CPM columns
                topTables_filtered = topTables[[gene_name] + new_column_names]
                # Step 4: Set 'Ensemble_gene_id' as index in table2_filtered
                topTables_filtered.set_index(gene_name, inplace=True)
				# Step 5: Transpose table2_filtered so that each sample ID becomes a row, and each gene becomes a column
                topTables_transposed = topTables_filtered.transpose()
				# Step 6: Join the health status from table1 with the transposed table2
                data = topTables_transposed.join(targets[illness_status_col])
                #st.write("First 4 rows of the dataframe:", data.head(4), use_container_width=True)
				
			
				# Check the resulting merged data structure
                #st.write("New dataframe:", data.head(), use_container_width=True)
				# Define how many columns to show from the start and end
                num_display_cols = 2
                num_display_rows = 4
                
				# Select the first `num_display_rows` rows, first `num_display_cols` columns, and last `num_display_cols` columns
                display_data = pd.concat([
                    data.iloc[:num_display_rows, :num_display_cols],  # First two columns of the first four rows
                    pd.DataFrame({f"...": ["..."] * num_display_rows}),  # Ellipsis to indicate skipped columns
                    data.iloc[:num_display_rows, -num_display_cols:]  # Last two columns of the first four rows
                ], axis=1)
				
				# Display the modified DataFrame in Streamlit
                st.write("First 4 rows of the new dataframe (with skipped columns in between):", display_data.head(4), use_container_width=True)
				
				
				# Optionally, store relevant_columns in a variable for further use
                # This could be done for further processing or machine learning steps

