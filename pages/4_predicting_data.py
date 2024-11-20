import streamlit as st
import numpy as np
import re
import pandas as pd

st.set_page_config(page_title="Predicting data")
st.markdown("# Predicting new data")
st.sidebar.header("Predicting New Data")
st.write(
    """Upload the wanted sample file(s) on which you want to predict the outcome of the illness with the previous created model. Keep in mind that the gene expressions used in the training data set should also be reoccurring in the newly uploaded dataset(s)."""
    )

if "model" in st.session_state and st.session_state["model"] is not None:
    model = st.session_state.model
    X_train_selected = st.session_state.X_train_selected
    min_max_scaler = st.session_state.min_max_scaler
    pca = st.session_state.pca
    
    #### INPUT AND PROCESSING OF THE DATA OF THE USER TO PREDICT THE ILLNESS
    max_files = 2
    uploaded_test_files = st.file_uploader("Upload CSV file with gene expressions for new patients", accept_multiple_files=True, type=["csv", "txt"])
    if len(uploaded_test_files) > max_files:
        st.error(f"Too many files uploaded! Please upload at most {max_files} files.")
    else:
        if uploaded_test_files:
           # Initialize file settings and placeholders
           test_file_settings = {}
           test_file_names = []  # List to store original filenames
           for idx, uploaded_test_file in enumerate(uploaded_test_files):
               test_file_name = uploaded_test_file.name  # Use the original filename
               test_file_names.append(test_file_name)  # Store the filename
               with st.expander(f"Settings for {test_file_name}", expanded=True):
                   # Settings for headers, separator, decimal indicator, etc.
                   has_header_test = st.checkbox(f"Header present in {test_file_name}", key=f"header_test{idx}", value=True)
                   separator_test = st.radio(f"Separator for {test_file_name}:", options=["Tab", "Comma", "Semicolon"], key=f"separator_test{idx}", horizontal=True)
                   sep_dict_test = {"Tab": "\t", "Comma": ",", "Semicolon": ";"}
                   sep_test = sep_dict_test[separator_test]
                   decimal_test = st.radio(f"Decimal indicator for {test_file_name}:", options=[".", ","], key=f"decimal_test{idx}", horizontal=True)

                   # Load file with user-defined settings
                   try:
                       df = pd.read_csv(uploaded_test_file, sep=sep_test, header=0 if has_header_test else None, decimal=decimal_test)
                       test_file_settings[test_file_name] = {"dataframe": df}
                       st.write(f"Preview of {test_file_name}")
                       st.write(df.head())
                   except Exception as e:
                       st.error(f"Error loading {test_file_name}: {e}")

    	    # Step 2: Data processing when the user uploaded only one file
           #if len(uploaded_files) == 1:
           #     with st.expander("Single File Processing", expanded=True):
           #         # Placeholder: Replace with the actual loaded DataFrame
           #         data = pd.DataFrame()  # file_settings[single_file_name]["dataframe"]
    #
           #         # Radio button to select orientation
           #         st.write("Is each sample data given in columns or rows?")
           #         sample_orientation = st.radio("Select orientation", options=["Sample data in columns", "Sample data in rows"])
    # DATA PREPROCESSING WHEN USER UPLOADS ONLY ONE FILE
           if len(uploaded_test_files) == 1:
               with st.expander("Single File Processing", expanded=True):
                   single_file_name = test_file_names[0]
                   data_one_file = test_file_settings[single_file_name]["dataframe"]

                   # Radio button for single selection between two options
                   st.write("Is each sample data given in columns or rows?")
                   sample_orientation = st.radio("Select orientation", options=["Sample data in columns", "Sample data in rows"])

                   if sample_orientation == "Sample data in rows":
                        st.write("Processing data where each sample is in a row:")

                        # Request user inputs for row-based identifiers
                        sample_id_col = st.text_input("Enter the column name that holds sample IDs:")

                        # Ensure that the input columns are in the DataFrame
                        if sample_id_col in data_one_file.columns:
                            # Let the user select the method for gene expression columns
                            selection_method = st.radio(
                                "Choose how to select the gene expression columns:",
                                options=["Use pattern matching", "Manually select first and last columns"]
                            )

                            # Initialize variable to hold gene expression columns
                            gene_expression_columns = []

                            # Process according to the user's selection
                            if selection_method == "Use pattern matching":
                                st.write("Pattern matching option is selected.")
                                pattern = st.text_input("Enter a common pattern in the column names for gene expression (e.g., 'CPM_'): ")
                                if pattern:
                                    gene_expression_columns = data_one_file.columns[data_one_file.columns.str.contains(pattern)]
                                    if not gene_expression_columns.empty:
                                        # Assuming `gene_expression_columns` is a list of column names
                                        gene_expression_columns_list = list(gene_expression_columns)  # Ensure it's a list

                                        # Display the first three, some ellipsis, and the last three
                                        first_three = gene_expression_columns_list[:3]
                                        last_three = gene_expression_columns_list[-3:]

                                        # Combine them with ellipsis in between
                                        display_columns = first_three + ['...'] + last_three

                                        # Create a DataFrame to show the selected columns
                                        gene_expression_df = pd.DataFrame(display_columns, columns=["Gene Expression Columns"])

                                        # Display the DataFrame
                                        st.write("Selected gene expression columns:")
                                        st.dataframe(gene_expression_df, height=280)  # Adjust the height to control scroll area
                                    else:
                                        st.error("No columns matched the provided pattern.")

                            elif selection_method == "Manually select first and last columns":
                                st.write("First and last column selection is selected.")
                                # Allow user to select the first and last columns
                                first_gene_col = st.selectbox("Select the first gene expression column:", data_one_file.columns)
                                last_gene_col = st.selectbox("Select the last gene expression column:", data_one_file.columns)

                                if first_gene_col and last_gene_col:
                                    # Get column indices and validate the order
                                    first_index = data_one_file.columns.get_loc(first_gene_col)
                                    last_index = data_one_file.columns.get_loc(last_gene_col)

                                    if first_index <= last_index:
                                        # Select all columns between the first and last specified columns
                                        gene_expression_columns = data_one_file.columns[first_index:last_index + 1]
                                        # Assuming `gene_expression_columns` is a list of column names
                                        gene_expression_columns_list = list(gene_expression_columns)  # Ensure it's a list

                                        # Display the first three, some ellipsis, and the last three
                                        first_three = gene_expression_columns_list[:3]
                                        last_three = gene_expression_columns_list[-3:]

                                        # Combine them with ellipsis in between
                                        display_columns = first_three + ['...'] + last_three

                                        # Create a DataFrame to show the selected columns
                                        gene_expression_df = pd.DataFrame(display_columns, columns=["Gene Expression Columns"])

                                        # Display the DataFrame
                                        st.write("Selected gene expression columns:")
                                        st.dataframe(gene_expression_df, height=280)  # Adjust the height to control scroll area
                                    else:
                                        st.error("The first column should be before the last column in the data structure.")

                            # Now create a new dataframe in the order: sample_id_col, gene expression columns, illness_status_col
                            if not gene_expression_columns.empty:  # Check if gene_expression_columns is not empty
                                # Extract the columns from the data
                                sample_ids = data_one_file[sample_id_col]

                                # Create the new dataframe with the required order
                                test_data = pd.concat([sample_ids, data_one_file[gene_expression_columns]], axis=1)
                                test_data = test_data.rename(columns={sample_id_col: "Sample_ID"})

                                # Display the new dataframe
     	    			            # Define how many columns to show from the start and end
                                num_display_cols = 2
                                num_display_rows = 4

    	    			            # Select the first `num_display_rows` rows, first `num_display_cols` columns, and last `num_display_cols` columns
                                display_data = pd.concat([
                                    test_data.iloc[:num_display_rows, :num_display_cols],  # First two columns of the first four rows
                                    pd.DataFrame({f"...": ["..."] * num_display_rows}),  # Ellipsis to indicate skipped columns
                                    test_data.iloc[:num_display_rows, -num_display_cols:]  # Last two columns of the first four rows
                                ], axis=1)                           
                                st.write("First 4 rows of the new dataframe (with skipped columns in between):", display_data.head(4), use_container_width=True)
                            else:
                                st.error("No gene expression columns were selected. Please check your selections.")

                            # Row-based workflow using the specified columns
                            sample_data = data_one_file.set_index(sample_id_col)
                        else:
                            st.error("One or both of the specified columns do not exist in the data.")

                   elif sample_orientation == "Sample data in columns":
                       st.write("Please convert your data first to where the samples' data is shown per row.")


    # DATA PREPROCESSING WHEN USER UPLOADS 2 FILES
           # Step 3: Data Processing when the user uploaded two files
           if len(uploaded_test_files) == 2:
               with st.expander("File Type Selection", expanded=True):
                   data_type_topTables = st.radio(f"Which file contains gene expression data?", test_file_names, key="gene_expression_radio")
                   data_type_targets = test_file_names[1] if data_type_topTables == test_file_names[0] else test_file_names[0]

                   # Reference the dataframes based on user selection
                   topTables = test_file_settings[data_type_topTables]["dataframe"]
                   targets = test_file_settings[data_type_targets]["dataframe"]

                   # Column inputs for illness status, sample IDs and gene_names
                   gene_names = st.text_input("Enter the column name in the gene expression table that holds the gene names or ID's:", key="gene_expression_text") 
                   sample_id_col = st.text_input("Enter the column name in sample data that holds sample IDs:", key="sample_id_text")

                   # Step 3: Data Processing - Find matching headers and rename them
                   if sample_id_col in targets.columns:    
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

                       ## Step 4: Create a new DataFrame with only the relevant columns
                       #cpm_columns = topTables[new_column_names].copy()  # Copy only the renamed columns
                       #relevant_columns = pd.concat([topTables[[gene_name]], cpm_columns], axis=1)  # Include the gene name column
                       ## Reset the index to remove the default index
                       #relevant_columns.reset_index(drop=True, inplace=True)
                       ##st.write("Relevant Columns After Adjustments:", relevant_columns.head(), use_container_width=True)
    	    			##ENDING ON TRYING TO DELETE THE GHOST COLUMN AT THE RELEVENT COLUMNS PART


                       # Step 2: Filter `topTables` to keep only the gene IDs and relevant CPM columns
                       topTables_filtered = topTables[[gene_names] + new_column_names]

                       # Step 3: Set `gene_names` as the index in `topTables_filtered`
                       topTables_filtered.set_index(gene_names, inplace=True)

                       # Step 4: Transpose `topTables_filtered` so that each sample ID becomes a row and each gene a column
                       gene_name_and_CPM = topTables_filtered.transpose()

                       # Step 5: Check for missing values and issue a warning if needed
                       if gene_name_and_CPM.isna().any().any():
                           st.warning("Warning: Some gene names did not match any sample IDs, resulting in empty cells.")

                       # Step 6: Combine only the sample names (Sample_ID) from `targets` with `gene_name_and_CPM`
                       # Reset the index to make the sample IDs a column in `gene_name_and_CPM`
                       test_data = gene_name_and_CPM.reset_index(inplace=True)
                       test_data = gene_name_and_CPM.rename(columns={'index': 'Sample_ID'})


    	    			# Check the resulting merged data structure
                       #st.write("New dataframe:", data.head(), use_container_width=True)
    	    			# Define how many columns to show from the start and end
                       num_display_cols = 2
                       num_display_rows = 4

    	    			# Select the first `num_display_rows` rows, first `num_display_cols` columns, and last `num_display_cols` columns
                       display_data = pd.concat([
                           test_data.iloc[:num_display_rows, :num_display_cols],  # First two columns of the first four rows
                           pd.DataFrame({f"...": ["..."] * num_display_rows}),  # Ellipsis to indicate skipped columns
                           test_data.iloc[:num_display_rows, -num_display_cols:]  # Last two columns of the first four rows
                       ], axis=1)

    	    			# Display the modified DataFrame in Streamlit
                       st.write("First 4 rows of the new dataframe (with skipped columns in between):", display_data.head(4), use_container_width=True)
                   else:
                        st.error("One or more of the specified columns do not exist in the database")

    	    			# Optionally, store relevant_columns in a variable for further use
                       # This could be done for further processing or machine learning steps
    # if 'data' in locals():

    if 'test_data' in locals():

        sample_ids = test_data.iloc[:, 0]  # Extract the sample ID column
        X_test = test_data.iloc[:, 1:]    # Exclude the sample ID column

        #select top n features. lets say 300.
        #you can modify the value and see how the performance of the model changes
        # n_features=300

        X_train_selected = st.session_state.X_train_selected
        selected_features = X_train_selected.columns  # Get selected feature names
        # In the prediction part
        test_data_selected = X_test[selected_features]
        #test_data_selected = X_test.iloc[:, selected_scores_indices]

       # Align test data to the training feature set
        if set(X_train_selected.columns) != set(test_data_selected.columns):
            st.warning("Feature names in the test data do not match those used during training. Aligning features...")

            # Reindex test data to match the training features
            test_data_selected = test_data_selected.reindex(columns=X_train_selected.columns, fill_value=0)
            st.write("Test data successfully re-aligned with training features.")
        else:
            st.write("Test data features match training data features. No alignment needed.")
        # Apply the scaler directly (already trained on the same feature set)
        test_data_scaled = min_max_scaler.transform(test_data_selected)

        # Apply PCA transformation if it was part of the pipeline
        test_data_pca = pca.transform(test_data_scaled)

        st.write("The data is succesfully gone through the Feature Selection, the Data Scaler and the PCA steps.")

        # Make predictions using the pre-trained model
        predictions = model.predict(test_data_pca)

        results = pd.DataFrame({'Sample_ID': sample_ids, 'Prediction': predictions})
        # Show the predictions
        st.write("Sample ID's + Predictions:")
        st.write(results)
    else:
        st.error("The prediction couldn't be done.")

else:
    st.error("Couldn't fetch the model from the machine learning part.")