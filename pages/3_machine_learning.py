import streamlit as st
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,recall_score,precision_score,f1_score 
from sklearn.metrics import roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns

from styles import load_css

st.set_page_config(page_title="Machine Learning")
st.markdown("# The machine learning part:")
st.sidebar.header("Machine Learning")
st.write(
    """This is the machine learning configuration page. Select the options you want to use for the machine learning part. Modify the options and parameters if needed. If the results aren't to your satisfaction after the model has been created, you can get the accuracy of the testing data to your desired heights with adjusting the parameters."""
    )

# Check if 'data' exists in session_state
if "data" in st.session_state and st.session_state["data"] is not None:
    data = st.session_state["data"]

    load_css()

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
        data = data.dropna(axis=1)
    fig, ax = plt.subplots()
    data['status'].value_counts().plot.bar(ax=ax)
    # Display the plot 
    st.write("Simple graph to show the data distribution:")
    st.pyplot(fig)
   
#    # Tooltip-rich options
#    options = {
#        "SMOTE": """
#        <p>
#            <span class="tooltip">SMOTE<span class="emoji">❓</span>
#                <span class="tooltiptext">SMOTE (Synthetic Minority Oversampling Technique) generates synthetic samples by interpolating between existing minority class samples.
#                Works well for both binary and multiclass data with moderate imbalance. 
#                It'is goes particularly useful when dealing with highly imbalanced data, as it helps to balance the class distribution, which can improve model performance by reducing bias toward the majority class.</span>
#            </span>
#        </p>
#        """,
#        "ADASYN": """
#        <p>
#            <span class="tooltip">ADASYN<span class="emoji">❓</span>
#                <span class="tooltiptext">ADASYN (Adaptive Synthetic Sampling) focuses on difficult-to-learn samples by generating more synthetic data in sparse areas of the minority class distribution. Useful for both binary and multiclass imbalances.
#                Suitable for highly imbalanced datasets or cases with hard-to-classify samples.
#                Works well for multiclass data, especially when decision boundaries need fine-tuning.
#                Do not use when the data is noisy.</span>
#            </span>
#        </p>
#        """,
#        "No": """
#        <p>
#            <span class="tooltip">No Oversampling <span class="emoji">❓</span>
#                <span class="tooltiptext">Choose this if you do not want to apply any oversampling technique. Recommended if your data is already balanced.</span>
#            </span>
#        </p>
#        """
#    }
    
#    # Render the radio button for selecting oversampling techniquea
#    st.markdown("Do you want to balance the data more with an oversampling technique?")
#    sample_distribution = st.radio(
#        label="Select an oversampling technique:",
#        options=list(options.keys())
#    )
#    
#    # Display the tooltip description based on the selected option
#    st.markdown(options[sample_distribution], unsafe_allow_html=True)
    
    sample_distribution = st.radio("Do you want to balance the data more with an oversampling technique?", options=["SMOTE", "ADASYN", "No"])

    if sample_distribution == "SMOTE":
        st.write("The data will be processed with the SMOTE (Synthetic Minority Oversampling Technique) algorithm. This technique generates synthetic samples by interpolating between existing minority class samples. Works well for both binary and multiclass data with moderate imbalance. It is particularly useful when dealing with highly imbalanced data, as it helps to balance the class distribution, which can improve model performance by reducing bias toward the majority class.") 
    elif sample_distribution == "ADASYN":
        st.write("The data will be processed with the ADASYN (Adaptive Synthetic Sampling) oversampling technique. This technique ocuses on difficult-to-learn samples by generating more synthetic data in sparse areas of the minority class distribution. Useful for both binary and multiclass imbalances. Suitable for highly imbalanced datasets or cases with hard-to-classify samples. Works well for multiclass data, especially when decision boundaries need fine-tuning. Do not use when the data is noisy.")
                   
    elif sample_distribution == "No":
        st.write("The data will not be processed with an oversampling technique. Keep in mind the data has to be balanced for the classification step.")


    # Storing the data in 'X' and 'y'
    X=data.iloc[:, 1:-1]
    y=data.iloc[:, -1]

    # Step 1: Mutual Information Feature Selection with slider
    st.header("Mutual Information-Based Feature Selection")
    
    st.markdown("""
    <p>
        <span class="tooltip">Select the number of features you want when performing Feature Selection: <span class="emoji">❓</span>
            <span class="tooltiptext">Feature selection is essential to reduce dimensionality and prevent overfitting. 
                Reducing the number of features (genes) to around 300 strikes a balance between computational efficiency and model performance. 
                Limiting features helps avoid noise and overfitting while ensuring the model focuses on the most relevant genes. 
                It also helps to maintain biological relevance, as fewer features can be more meaningful for interpretation. 
                Studies suggest that focusing on a smaller subset of features enhances model efficiency and interpretability but may lead to underfitting if too many important features are excluded.
            </span>
        </span>
    </p>
    """, unsafe_allow_html=True)
    n_features = st.slider(
    label="",
    min_value=1,
    max_value=X.shape[1],
    value=min(300, X.shape[1]),  # Default to 20 or max available features
    step=25
    )
    #### Automatic method
    #let's encode target labels (y) with values between 0 and n_classes-1.
    #encoding will be done using the LabelEncoder
    label_encoder=LabelEncoder()
    label_encoder.fit(y)
    y_encoded=label_encoder.transform(y)
    labels=label_encoder.classes_
    st.session_state.labels = labels
    classes=np.unique(y_encoded)
    n_classes = len(classes)
    st.session_state.classes = classes
    st.write("The automatic label encoder step was succesfull:")
    st.write(f"The labels are {labels} and the classes are {classes}.")
    
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
    st.session_state.min_max_scaler = min_max_scaler
    st.write("The Normilazation step was successful.")

    # Principal compon  ent analysis steps
    pca = PCA(n_components=8)  # Adjust n_components based on explained variance
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    st.session_state.pca = pca
    st.session_state.X_train_pca = X_train_pca

    st.write("The Principal Component Analysis was succcessful.")

    if sample_distribution == "SMOTE":
        # transform the dataset
        oversample = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = oversample.fit_resample(X_train_pca, y_train)
        # summarize the new class distribution
        counter = Counter(y_train_resampled)
        st.write("The Oversampling Technique (SMOTE) was successful, the new class distribution is:")
        counter_dict = {str(key): int(value) for key, value in counter.items()}  # Convert numpy.int64 to int
        # Plotting the new class distribution
        fig, ax = plt.subplots()
        sns.barplot(x=list(counter_dict.keys()), y=list(counter_dict.values()), ax=ax)
        ax.set_title("Class Distribution After SMOTE")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Number of Samples")
        st.pyplot(fig)

    elif sample_distribution == "ADASYN":
        oversample = ADASYN(random_state=42)
        X_train_resampled, y_train_resampled = oversample.fit_resample(X_train_pca, y_train)
        # summarize the new class distribution
        counter = Counter(y_train_resampled)
        st.write("The Oversampling Technique (ADASYN) was successful, the new class distribution is:")
        counter_dict = {str(key): int(value) for key, value in counter.items()}  # Convert numpy.int64 to int
        # Plotting the new class distribution
        fig, ax = plt.subplots()
        sns.barplot(x=list(counter_dict.keys()), y=list(counter_dict.values()), ax=ax)
        ax.set_title("Class Distribution After ADASYN")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Number of Samples")
        st.pyplot(fig)

    elif sample_distribution == "No":
        X_train_resampled = X_train_pca
        y_train_resampled = y_train

    st.session_state.y_train_resampled = y_train_resampled
    st.session_state.X_train_resampled = X_train_resampled

    st.write("What type of classification algorithm do you want to use?")
    Classification_type = st.radio("Select classification type:", options=["Logistic Regression (L1)", "Random Forest Classifier", "XGBoost", "One Vs Rest Classifier"])
    st.session_state.Classification_type = Classification_type

    if Classification_type == "Logistic Regression (L1)":
        st.write("Choose this classification type when you want a simple and interpretable model. Your dataset has a linear relationship between features and the target. The data you are working with is high-dimensional (e.g., gene expression datasets) and want to perform feature selection, as L1 regularization (Lasso) shrinks irrelevant feature weights to zero. Best used when the dataset size is moderate to large, as logistic regression generally performs well with sufficient data. Keep in mind that this model doesn't work well with non-linear or complex data or data that is highly imbalanced.")

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
        conf_matrix = (confusion_matrix(y_true=y_test,y_pred=y_pred))
 
        
        # Show the accuracy, recall, precision and f1-score for the test set
        # Note, sometimes you need to supply a positive label (if not working with 0 and 1)
        # supply this with "pos_label='label'", in this case, the malign samples are the positives
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()


    elif Classification_type == "Random Forest Classifier":
        st.write("Choose this classification type when your data has complex relationships or non-linearities, as Random Forest captures interactions between features without requiring explicit modeling. If you model needs to be robust to noise and handles both categorical and continuous data well. Imbalanced data isn't a problem for this dataset. Keep in mind this model doesn't do well with small datasets, as it may overfit with insufficient data.")
        st.markdown("""
        <p>
            <span class="tooltip">Select the number of trees (n_estimators) for Random Forest: <span class="emoji">❓</span>
                <span class="tooltiptext">The n_estimators parameter specifies the number of decision trees in the Random Forest. Each tree votes on the output, and the majority vote determines the final prediction.
                When you lower the amount of trees, you make the model faster to train but may reduce accuracy and stability since fewer votes determine the output.
                When you higher the amount of trees, you increase accuracy and reduce variance by averaging more predictions, but they also make the model slower to train and require more memory.
                </span>
            </span>
        </p>
        """, unsafe_allow_html=True)

        n_estimators = st.slider(
            label="",
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
        # Check if you have over- or underfitting of your model by comparing the score of the training and test set
        st.write("Accuracy on the training set:",model.score(X_train_resampled,y_train_resampled))
        st.write("Accuracy on the test set:",model.score(X_test_pca,y_test))

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Extract feature importances from the random forest model
        feature_importances = model.feature_importances_
        st.session_state.feature_importances = feature_importances

        ## Create a DataFrame of feature importances
        #feature_importances_df = pd.DataFrame({
        #    "Feature": [f"PC{i+1}" for i in range(X_train_pca.shape[1])],
        #    "Importance": feature_importances
        #}).sort_values(by="Importance", ascending=False)
        #st.session_state.pca_options = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]

        ## Display top features in a table
        #st.table(feature_importances_df.head(20))  # Show top 20 features
#
        ## Visualize the top features in a bar plot
        #plt.figure(figsize=(10, 6))
        #plt.barh(feature_importances_df['Feature'][:20], feature_importances_df['Importance'][:20])  # top 20 features
        #plt.xlabel('Feature Importance')
        #plt.title('Random Forest Feature Importances')
        #st.pyplot(plt)

    elif Classification_type == "XGBoost":
        st.write("Choose this classification type when you are working")
        # Initialize XGBoost Classifier
        st.markdown("""
        <p>
            <span class="tooltip">Select the number of trees (n_estimators) for Random Forest: <span class="emoji">❓</span>
                <span class="tooltiptext">The n_estimators parameter specifies the number of boosting rounds in XGBoost. Each round builds on the errors of the previous model to reduce bias and variance.
        Lower values may lead to underfitting, as the model does not learn enough from the data.
        Higher values can improve performance but may increase training time and risk overfitting. Adjust based on the dataset size and complexity.
                </span>
            </span>
        </p>
        """, unsafe_allow_html=True)

        n_estimators = st.slider(
            label="",
            min_value=1,
            max_value=500,
            value=100,  # Default value set to 100
            step=10
        )   
        if n_classes == 2:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,          # Number of trees
                max_depth=6,               # Maximum depth of a tree
                learning_rate=0.1,         # Step size shrinkage
                objective='binary:logistic', # Objective function for binary classification
                random_state=42,           # For reproducibility
                eval_metric='logloss'      # Evaluation metric
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,          # Number of trees
                max_depth=6,               # Maximum depth of a tree
                learning_rate=0.1,         # Step size shrinkage
                objective='multi:softprob', # Objective function for binary classification
                random_state=42,           # For reproducibility
                eval_metric='logloss'      # Evaluation metric
            )            
        # Train the model
        model.fit(X_train_resampled, y_train_resampled)

        # Make predictions
        y_pred = model.predict(X_test_pca)

        st.write("Accuracy on the training set:", model.score(X_train_resampled,y_train_resampled))
        st.write("Accuracy on the test set:", model.score(X_test_pca,y_test))        

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

    elif Classification_type == "One Vs Rest Classifier":
        st.write("Choose this classification type when you are working with multiple classes and want to break the classification problem into multiple binary tasks. Keep in mind that this classifier doesn't do well with imbalanced data across classes.")
        model = OneVsRestClassifier(RandomForestClassifier(max_features=0.2, random_state=42))
        model.fit(X_train_resampled,y_train_resampled)
        

        y_pred = model.predict(X_test_pca)
        pred_prob = model.predict_proba(X_test_pca)

        # Training and test accuracy
        st.write("Accuracy on the training set:", model.score(X_train_resampled, y_train_resampled))
        st.write("Accuracy on the test set:", model.score(X_test_pca, y_test))

        # Confusion matrix
        conf_matrix = (confusion_matrix(y_true=y_test, y_pred=y_pred))

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

    if n_classes == 2:
        y_pred_proba = model.predict_proba(X_test_pca)[:, 1]

    else:
        y_pred_proba = model.predict_proba(X_test_pca)
        
    gene_names = X_train.columns[selected_scores_indices]
    st.session_state.X_train_selected = X_train_selected 
    st.session_state.conf_matrix = conf_matrix
    st.session_state.report_df = report_df
    st.session_state.y_pred_proba = y_pred_proba
    st.session_state.y_test = y_test
    st.session_state.gene_names = gene_names
    st.session_state.model = model
    st.session_state.X_test_pca = X_test_pca

    pca_components = pca.components_
    st.session_state.pca_components = pca_components

    # st.write(selected_scores_indices)
    # st.write(gene_names)

else:
    st.error("The data is not ready for machine learning. Please upload and process your data on the previous page.")

