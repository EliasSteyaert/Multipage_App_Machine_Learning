import streamlit as st
import pandas as pd
import re

# Streamlit app title
st.title("Gene Expression Analysis App")

# File uploader
uploaded_files = st.file_uploader("Upload your files (one or two)", accept_multiple_files=True, type=["txt", "csv"])

if uploaded_files:
    file_settings = {}

    for idx, uploaded_file in enumerate(uploaded_files):
        file_name = f"File {idx + 1}"

        # Expander to make settings for each file collapsible
        with st.expander(f"Settings for {file_name}", expanded=True):
            # Header option
            has_header = st.checkbox(f"Header present in {file_name}", key=f"header_{idx}")
            
            # Separator options
            separator = st.radio(
                f"Separator for {file_name}:",
                options=["Tab", "Comma", "Semicolon"],
                key=f"separator_{idx}",
                horizontal=True
            )
            sep_dict = {"Tab": "\t", "Comma": ",", "Semicolon": ";"}
            sep = sep_dict[separator]

            # Decimal indicator options
            decimal = st.radio(
                f"Decimal indicator for {file_name}:",
                options=[",", "."],
                key=f"decimal_{idx}",
                horizontal=True
            )

            # Load file into a DataFrame with user settings
            try:
                df = pd.read_csv(
                    uploaded_file,
                    sep=sep,
                    header=0 if has_header else None,
                    decimal=decimal
                )
                
                # Save file settings and DataFrame
                file_settings[file_name] = {
                    "dataframe": df,
                    "header": has_header,
                    "separator": sep,
                    "decimal": decimal
                }
                
                # Display the DataFrame head for confirmation
                st.write(f"Preview of {file_name}")
                st.write(df.head())

            except Exception as e:
                st.error(f"Error loading {file_name}: {e}")

    # Identifying files as gene expression or sample data
    # Step 2: User Input for Column Names and File Identification
    if len(uploaded_files) == 2:
        with st.expander("File Type Selection", expanded=True):
            data_type_topTables = st.radio("Which file contains gene expression data ('topTables')?", ["File 1", "File 2"])
            data_type_targets = "File 2" if data_type_topTables == "File 1" else "File 1"
            
            # Reference the dataframes based on user selection
            topTables = file_settings[data_type_topTables]["dataframe"]
            targets = file_settings[data_type_targets]["dataframe"]
            
            # Column inputs for illness status and sample IDs
            illness_status_col = st.text_input("Enter the column name in sample data that indicates patient illness status:")
            sample_id_col = st.text_input("Enter the column name in sample data that holds sample IDs:")

            # Step 3: Data Processing - Find matching headers
            if illness_status_col and sample_id_col:
                # Replace hyphens with underscores in the sample ID column of the targets DataFrame
                targets[sample_id_col] = targets[sample_id_col].str.replace('-', '_')
                # Extract sample IDs from targets DataFrame
                sample_ids = targets[sample_id_col].tolist()  # Make sure to use the correct column

                # Initialize a list to hold matching headers
                rename_dict = []

                # Loop through sample IDs and find corresponding headers in topTables
                for sample_id in sample_ids:
                    # Create a regex pattern that allows for any characters before and after the sample ID
                    pattern = re.compile(f".*{re.escape(sample_id)}.*")

                    # Check each header in the topTables DataFrame
                    for header in topTables.columns:
                        if pattern.match(header):
                            #Map old header name to the new sample ID
                            rename_dict[header] = sample_id

                # Rename the columns in topTables based on the rename_dict
                topTables.rename(columns=rename_dict, inplace=True)

                # Display the results
                st.write("Updated topTables headers:", topTables.columns.tolist())

                # Continue with further processing as needed
            
			# Confirming selection
            st.write("You have selected the following settings:")
            st.write("Gene expression data file:", data_type_topTables)
            st.write("Sample data file:", data_type_targets)
            st.write("Illness status column:", illness_status_col)
            
	#Data preprocessing
