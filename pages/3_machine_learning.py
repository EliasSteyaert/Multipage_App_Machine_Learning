import streamlit as st
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,recall_score,precision_score,f1_score 
from sklearn.metrics import roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


st.set_page_config(page_title="Machine Learning")
st.markdown("# The machine learning part:")
st.sidebar.header("Machine Learning")
st.write(
    """Select the options you want to use for the machine learning part. Modify the options if needed afterwards to get the accuracy of the testing data to your desired heights."""
    )

# Check if 'data' exists in session_state
if "data" in st.session_state and st.session_state["data"] is not None:
    data = st.session_state["data"]

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
    st.session_state.X_train_selected = X_train_selected

    X_test_selected=X_test.iloc[:, selected_scores_indices]

    st.write("The Feature Selection step was successful.")

    # scale data between 0 and 1
    min_max_scaler = MinMaxScaler()
    X_train_norm = min_max_scaler.fit_transform(X_train_selected)
    X_test_norm = min_max_scaler.transform(X_test_selected)
    st.session_state.min_max_scaler = min_max_scaler
    st.write("The Normilazation step was successful.")

    # Principal component analysis steps
    pca = PCA(n_components=8)  # Adjust n_components based on explained variance
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    st.session_state.pca = pca

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
        st.session_state.model = model
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
        st.session_state.model = model

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
        model.fit(X_train_resampled,y_train_resampled)
        st.session_state.model = model

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
    st.error("The data is not ready for machine learning. Please upload and process your data on the previous page.")

