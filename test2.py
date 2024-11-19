#IMPORTING THE NEEDED MODULES
import streamlit as st
import pandas as pd
import re
import numpy as np  

import matplotlib.pyplot as plt

#preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from sklearn.feature_selection import mutual_info_classif
import imblearn
from imblearn.over_sampling import SMOTE
from collections import Counter

#classification
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,recall_score,precision_score,f1_score 
from sklearn.metrics import roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
# Step 1: File Uploads and Initial Settings

# GENERAL PREPROCESSING PART
uploaded_files = st.file_uploader("Upload your files (one or two)", accept_multiple_files=True, type=["txt", "csv"])

# Set a maximum file limit
max_files = 2  # Adjust this to your desired limit

# Check if the number of uploaded files exceeds the limit
if len(uploaded_files) > max_files:
    st.error(f"Too many files uploaded! Please upload at most {max_files} files.")
else:
    if uploaded_files:
       # Initialize file settings and placeholders
       file_settings = {}
       file_names = []  # List to store original filenames
       for idx, uploaded_file in enumerate(uploaded_files):
           file_name = uploaded_file.name  # Use the original filename
           file_names.append(file_name)  # Store the filename
           with st.expander(f"Settings for {file_name}", expanded=True):
               # Settings for headers, separator, decimal indicator, etc.
               has_header = st.checkbox(f"Header present in {file_name}", key=f"header_{idx}", value=True)
               separator = st.radio(f"Separator for {file_name}:", options=["Tab", "Comma", "Semicolon"], key=f"separator_{idx}", horizontal=True)
               sep_dict = {"Tab": "\t", "Comma": ",", "Semicolon": ";"}
               sep = sep_dict[separator]
               decimal = st.radio(f"Decimal indicator for {file_name}:", options=[".", ","], key=f"decimal_{idx}", horizontal=True)

               # Load file with user-defined settings
               try:
                   df = pd.read_csv(uploaded_file, sep=sep, header=0 if has_header else None, decimal=decimal)
                   file_settings[file_name] = {"dataframe": df}
                   st.write(f"Preview of {file_name}")
                   st.write(df.head())
               except Exception as e:
                   st.error(f"Error loading {file_name}: {e}")

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
       if len(uploaded_files) == 1:
           with st.expander("Single File Processing", expanded=True):
               single_file_name = file_names[0]
               data_one_file = file_settings[single_file_name]["dataframe"]

               # Radio button for single selection between two options
               st.write("Is each sample data given in columns or rows?")
               sample_orientation = st.radio("Select orientation", options=["Sample data in columns", "Sample data in rows"])

               if sample_orientation == "Sample data in columns":
                   st.write("Processing data where each sample is in a column:")
                   

               elif sample_orientation == "Sample data in rows":
                    st.write("Processing data where each sample is in a row:")

                    # Request user inputs for row-based identifiers
                    illness_status_col = st.text_input("Enter the column name that indicates patient illness status:")
                    sample_id_col = st.text_input("Enter the column name that holds sample IDs:")

                    # Ensure that the input columns are in the DataFrame
                    if illness_status_col in data_one_file.columns and sample_id_col in data_one_file.columns:
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
                            illness_status = data_one_file[illness_status_col]

                            # Create the new dataframe with the required order
                            data = pd.concat([sample_ids, data_one_file[gene_expression_columns], illness_status], axis=1)
                            data = data.rename(columns={sample_id_col: "Sample_ID", illness_status_col: "status"})

                            # Display the new dataframe
 	    			            # Define how many columns to show from the start and end
                            num_display_cols = 2
                            num_display_rows = 4

	    			            # Select the first `num_display_rows` rows, first `num_display_cols` columns, and last `num_display_cols` columns
                            display_data = pd.concat([
                                data.iloc[:num_display_rows, :num_display_cols],  # First two columns of the first four rows
                                pd.DataFrame({f"...": ["..."] * num_display_rows}),  # Ellipsis to indicate skipped columns
                                data.iloc[:num_display_rows, -num_display_cols:]  # Last two columns of the first four rows
                            ], axis=1)                           
                            st.write("First 4 rows of the new dataframe (with skipped columns in between):", display_data.head(4), use_container_width=True)
                        else:
                            st.error("No gene expression columns were selected. Please check your selections.")

                        # Row-based workflow using the specified columns
                        illness_status_data = data_one_file[illness_status_col]
                        sample_data = data_one_file.set_index(sample_id_col)
                    else:
                        st.error("One or both of the specified columns do not exist in the data.")
# DATA PREPROCESSING WHEN USER UPLOADS 2 FILES
       # Step 3: Data Processing when the user uploaded two files
       if len(uploaded_files) == 2:
           with st.expander("File Type Selection", expanded=True):
               data_type_topTables = st.radio(f"Which file contains gene expression data?", file_names)
               data_type_targets = file_names[1] if data_type_topTables == file_names[0] else file_names[0]

               # Reference the dataframes based on user selection
               topTables = file_settings[data_type_topTables]["dataframe"]
               targets = file_settings[data_type_targets]["dataframe"]

               # Column inputs for illness status, sample IDs and gene_names
               gene_names = st.text_input("Enter the column name in the gene expression table that holds the gene names or ID's:")
               illness_status_col = st.text_input("Enter the column name in sample data that indicates patient illness status:")
               sample_id_col = st.text_input("Enter the column name in sample data that holds sample IDs:")

               # Step 3: Data Processing - Find matching headers and rename them
               if illness_status_col in targets.columns and sample_id_col in targets.columns:    
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


	    			#converting to one "data" dataframe
                   # Step 1: Set the 'Sample_ID' as index in `table1`
                   targets.set_index(sample_id_col, inplace=True)
                   # Step 2: Filter topTables to keep only the gene IDs and relevant CPM columns
                   topTables_filtered = topTables[[gene_names] + new_column_names]
                   # Step 4: Set 'Ensemble_gene_id' as index in table2_filtered
                   topTables_filtered.set_index(gene_names, inplace=True)
	    			# Step 5: Transpose table2_filtered so that each sample ID becomes a row, and each gene becomes a column
                   gene_name_and_CPM = topTables_filtered.transpose()
                   if gene_name_and_CPM.isna().any().any():
                        st.warning("Warning: Some gene names did not match any sample IDs, resulting in empty cells.")
	    			# Step 6: Join the health status from table1 with the transposed table2
                   data = gene_name_and_CPM.join(targets[illness_status_col])
                   # Rename the columns for sample ID and illness status
                   data = data.rename(columns={sample_id_col: "Sample_ID", illness_status_col: "status"})
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
               else:
                    st.error("One or more of the specified columns do not exist in the database")

	    			# Optionally, store relevant_columns in a variable for further use
                   # This could be done for further processing or machine learning steps
                    
# MACHINE LEARNING PART
# Deleting possible NA values and plotting a graph of the data distribution
if 'data' in locals():
    st.write("Starting the machine learning process...")
    na_counts = data.isna().sum()
    total_na_values = na_counts.sum()  # Sum of all NA values across the DataFrame

    # Check if there are any NA values in the DataFrame
    if total_na_values == 0:
        st.write("No NA values found.")
    else:
        # Show the count of NA values and display the columns with them
        st.write(f"Found {total_na_values} NA values across {na_counts[na_counts > 0].count()} columns.")
        st.write("Removing columns with NA values...")
        
        # Drop columns with any NA values
        data_cleaned = data.dropna(axis=1)
    fig, ax = plt.subplots()
    data['status'].value_counts().plot.bar(ax=ax)
    # Display the plot 
    st.write("Simple graph to show the data distribution:")
    st.pyplot(fig)
    
    # Radio button for single selection between two options
    st.write("Is the data distributed?")
    sample_distribution = st.radio("Do you want to balance the data more with an oversampling technique (SMOTE)", options=["Yes", "No"])

    if sample_distribution == "Yes":
        st.write("The data will be processed with the oversampling technique.")
                   

    elif sample_distribution == "No":
        st.write("The data will not be processed with the oversampling technique. Keep in mind the data has to be balanced for the classification step.")


    # Storing the data in 'X' and 'y'
    X=data.iloc[:, 1:-1]
    y=data.iloc[:, -1]

    # Step 1: Mutual Information Feature Selection with slider
    st.header("Mutual Information-Based  Feature Selection")
    n_features = st.slider(
    "Select the number of features you want when performing Feature Selection",
    min_value=1,
    max_value=X.shape[1],
    value=min(300, X.shape[1])  # Default to 20 or max available features
    )
    #### Automatic method
    #let's encode target labels (y) with values between 0 and n_classes-1.
    #encoding will be done using the LabelEncoder
    label_encoder=LabelEncoder()
    label_encoder.fit(y)
    y_encoded=label_encoder.transform(y)
    labels=label_encoder.classes_
    classes=np.unique(y_encoded)
    st.write("The automatic label encoder step was succesfull:")
    st.write(f"The lablels are {labels} and the classes are {classes}.")
    
    #split data into training and test sets
    X_train,X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.2,random_state=42)

    MI=mutual_info_classif(X_train,y_train)
    #select top n features. lets say 300.
    #you can modify the value and see how the performance of the model changes
    # n_features=300
    selected_scores_indices=np.argsort(MI)[::-1][0:n_features]
    
    X_train_selected=X_train.iloc[:, selected_scores_indices]
    X_test_selected=X_test.iloc[:, selected_scores_indices]

    st.write("The Feature Selection step was successful.")

    # scale data between 0 and 1
    min_max_scaler = MinMaxScaler()
    X_train_norm = min_max_scaler.fit_transform(X_train_selected)
    X_test_norm = min_max_scaler.transform(X_test_selected)

    st.write("The Normilazation step was successful.")

    # Principal component analysis steps
    pca = PCA(n_components=8)  # Adjust n_components based on explained variance
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    st.write("The Principal Component Analysis was succcessful.")

    if sample_distribution == "Yes":
        # transform the dataset
        oversample = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = oversample.fit_resample(X_train_pca, y_train)
        # summarize the new class distribution
        counter = Counter(y_train_resampled)
        st.write("The Oversampling Technique (SMOTE) was successful, the new class distribution is:")
        counter_dict = {str(key): int(value) for key, value in counter.items()}  # Convert numpy.int64 to int
        st.write("The data distribution after the SMOTE technique:")
        # Plotting the new class distribution
        fig, ax = plt.subplots()
        sns.barplot(x=list(counter_dict.keys()), y=list(counter_dict.values()), ax=ax)
        ax.set_title("Class Distribution After SMOTE")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Number of Samples")
        st.pyplot(fig)
    

    elif sample_distribution == "No":
        X_train_resampled = X_train_pca
        y_train_resampled = y_train


    st.write("What type of classification algorithm do you want to use?")
    Classification_type = st.radio("Select classification type", options=["Logistic Regression (L1)", "Random Forest Classifier", "One Vs Rest Classifier"])

    if Classification_type == "Logistic Regression (L1)":
        st.write("Processing the data with the Logistic Regression:")
        # Initialize a logistic regression model, fit the data. Start with a C-value of 1
        model= LogisticRegression(C=1, class_weight='balanced', penalty='l1', solver='liblinear')
        model.fit(X_train_resampled,y_train_resampled)
        y_pred = model.predict(X_test_pca)

        # Check if you have over- or underfitting of your model by comparing the score of the training and test set
        st.write("Accuracy on the training set:",model.score(X_train_resampled,y_train_resampled))
        st.write("Accuracy on the test set:",model.score(X_test_pca,y_test))

        # Predict values for the test sete
        y_pred=model.predict(X_test_pca)                       
        # Look at the confusion matrix, what do the different values mean in this case?
        # Hint: if you don't know the syntax/meaning for a specific funtion, you can always look this up
        # in jupyter notebook by executing "?function_name"
        st.write("Confusion matrix:")
        st.table(confusion_matrix(y_true=y_test,y_pred=y_pred))
        st.divider()
        
        # Show the accuracy, recall, precision and f1-score for the test set
        # Note, sometimes you need to supply a positive label (if not working with 0 and 1)
        # supply this with "pos_label='label'", in this case, the malign samples are the positives
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.table(report_df)

    elif Classification_type == "Random Forest Classifier":
        st.write("Processing the data with the Random Forest Classifier:")

        n_estimators = st.slider(
            "Select the number of trees (n_estimators) for Random Forest",
            min_value=10,
            max_value=500,
            value=100,  # Default value set to 100
            step=10
        )   

        # Initialize the Random Forest model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        # Train the model on the resampled (or original) training data
        model.fit(X_train_resampled, y_train_resampled)

        # Make predictions on the test data
        y_pred = model.predict(X_test_pca)

        # Evaluate the model
        accuracy_train = accuracy_score(y_train_resampled, model.predict(X_train_resampled))
        accuracy_test = accuracy_score(y_test, y_pred)

        # Display the results
        st.write("Random Forest Classifier Results:")
        st.write(f"Accuracy on the training set: {accuracy_train:.4f}")
        st.write(f"Accuracy on the test set: {accuracy_test:.4f}")

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        st.write("Confusion Matrix:")
        st.table(conf_matrix)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("Classification Report:")
        st.table(report_df)

        # Feature Importances (Optional)
        st.write("Feature Importances:")

        # Extract feature importances from the random forest model
        feature_importances = model.feature_importances_

        # Create a DataFrame of feature importances
        feature_importances_df = pd.DataFrame({
            "Feature": [f"PC{i+1}" for i in range(X_train_pca.shape[1])],
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)

        # Display top features in a table
        st.table(feature_importances_df.head(20))  # Show top 20 features

        # Visualize the top features in a bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances_df['Feature'][:20], feature_importances_df['Importance'][:20])  # top 20 features
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importances')
        st.pyplot(plt)

    elif Classification_type == "One Vs Rest Classifier":
        st.write("Processing the data with the One VS Rest Classifier:")
        model = OneVsRestClassifier(RandomForestClassifier(max_features=0.2, random_state=42))
        model = model.fit(X_train_resampled,y_train_resampled)
        y_pred = model.predict(X_test_pca)
        pred_prob = model.predict_proba(X_test_pca)

        # Training and test accuracy
        st.write("Accuracy on the training set:", model.score(X_train_resampled, y_train_resampled))
        st.write("Accuracy on the test set:", model.score(X_test_pca, y_test))

        # Confusion matrix
        st.write("Confusion matrix:")
        st.table(confusion_matrix(y_true=y_test, y_pred=y_pred))

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("Classification report:")
        st.table(report_df)



else:
    st.error("The data is not ready for machine learning. Make sure all the necessary selections are made.")


#### INPUT AND PROCESSING OF THE DATA OF THE USER TO PREDICT THE ILLNESS
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