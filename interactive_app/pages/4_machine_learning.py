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

import re

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns   

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import learning_curve

from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import graphviz
import shap

from streamlit_extras.switch_page_button import switch_page
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
    y_train_resampled = st.session_state.y_train_resampled
    X_train_resampled = st.session_state.X_train_resampled
    classes = st.session_state.classes
    labels = st.session_state.labels
    gene_names = st.session_state.gene_names
    y_test = st.session_state.y_test
    pca_components = st.session_state.pca_components
    n_components = st.session_state.n_components
    X_test_pca = st.session_state.X_test_pca
    X_train_selected = st.session_state.X_train_selected

    n_classes = len(classes)

    load_css()

    # Tooltip explanation
    tooltip_text = """
    When clicked upon one of the options, underneath shall the explanation be shown.
    """
    # Add a tooltip to the radio button
    st.markdown("""
        <p style="margin-top: 20px; margin-bottom: -35px;">
            <span class="tooltip">
                <span class="emoji">❓</span>
                What type of <b>classification algorithm</b> do you want to use?
                <span class="tooltiptext">{}</span>
            </span>
        </p>
        """.format(tooltip_text), unsafe_allow_html=True)

    # Radio button for the classification type
    Classification_type = st.radio("", options=["Logistic Regression (L1)", "Random Forest Classifier", "XGBoost", "One Vs Rest Classifier"])

    if Classification_type == "Logistic Regression (L1)":
        st.write("Choose this classification type when you want a simple and interpretable model. Your dataset has a linear relationship between features and the target. The data you are working with is high-dimensional (e.g., gene expression datasets) and you want to perform feature selection, as L1 regularization (Lasso) shrinks irrelevant feature weights to zero. Best used when the dataset size is moderate to large, as logistic regression generally performs well with sufficient data. Keep in mind that this model doesn't work well with non-linear or complex data or data that is highly imbalanced.")
        st.write("")
        st.write("")
        # Initialize a logistic regression model, fit the data. Start with a C-value of 1
        model= LogisticRegression(C=1, class_weight='balanced', penalty='l1', solver='liblinear')
        model.fit(X_train_resampled,y_train_resampled)
        y_pred = model.predict(X_test_pca)


        # Evaluate the model
        accuracy_train = accuracy_score(y_train_resampled, model.predict(X_train_resampled))
        accuracy_test = accuracy_score(y_test, y_pred)

        # Display the results
        st.write("Logistic Regression (L1) Classifier Results:")
        # Check if you have over- or underfitting of your model by comparing the score of the training and test set
        st.write("Explicit Accuracy on the training set:", accuracy_train)
        st.write("Explicit Accuracy on the test set:", accuracy_test)

        conf_matrix = (confusion_matrix(y_true=y_test,y_pred=y_pred))
 
        # Show the accuracy, recall, precision and f1-score for the test set
        # Note, sometimes you need to supply a positive label (if not working with 0 and 1)
        # supply this with "pos_label='label'", in this case, the malign samples are the positives
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()


    elif Classification_type == "Random Forest Classifier":
        st.write("Choose this classification type when your data has complex relationships or non-linearities, as Random Forest captures interactions between features without requiring explicit modeling. If you model needs to be robust to noise and handles both categorical and continuous data well. Imbalanced data isn't a problem for this dataset. Keep in mind this model doesn't do well with small datasets, as it may overfit with insufficient data.")
        st.write("")
        st.write("")      
        
        st.markdown("""
        <p style="margin-top: 0px; margin-bottom: -35px;">
            <span class="tooltip"><span class="emoji">❓</span>Select the <b>number of trees</b> (n_estimators) for Random Forest: 
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
        st.write("Explicit Accuracy on the training set:", accuracy_train)
        st.write("Explicit Accuracy on the test set:", accuracy_test)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Extract feature importances from the random forest model
        pca_importances = model.feature_importances_
        # Map to original features using PCA components
        original_importances = np.dot(pca_components.T, pca_importances)

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
        st.write("Choose this classification type when you want a powerful and efficient model that excels with complex and high-dimensional data. XGBoost (Extreme Gradient Boosting) is particularly well-suited for classification tasks where accuracy is crucial, and the relationships between features and target are non-linear.")
        st.write("")
        st.write("")

        # Initialize XGBoost Classifier
        st.markdown("""
        <p style="margin-top: 0px; margin-bottom: -35px;">
            <span class="tooltip"><span class="emoji">❓</span>Select the <b>number of trees</b> (n_estimators) for Random Forest: 
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

        # Evaluate the model
        accuracy_train = accuracy_score(y_train_resampled, model.predict(X_train_resampled))
        accuracy_test = accuracy_score(y_test, y_pred)

        # Display the results
        st.write("XGBoost Classifier Results:")
        # Check if you have over- or underfitting of your model by comparing the score of the training and test set
        st.write("Explicit Accuracy on the training set:", accuracy_train)
        st.write("Explicit Accuracy on the test set:", accuracy_test)      

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

    elif Classification_type == "One Vs Rest Classifier":
        st.write("Choose this classification type when you are working with multiple classes and want to break the classification problem into multiple binary tasks. Keep in mind that this classifier doesn't do well with imbalanced data across classes.")
        st.write("")
        st.write("")

        model = OneVsRestClassifier(RandomForestClassifier(max_features=0.2, random_state=42))
        model.fit(X_train_resampled,y_train_resampled)
        

        y_pred = model.predict(X_test_pca)
        pred_prob = model.predict_proba(X_test_pca)

        # Evaluate the model
        accuracy_train = accuracy_score(y_train_resampled, model.predict(X_train_resampled))
        accuracy_test = accuracy_score(y_test, y_pred)

        # Display the results
        st.write("Random Forest Classifier Results:")
        # Check if you have over- or underfitting of your model by comparing the score of the training and test set
        st.write("Explicit Accuracy on the training set:", accuracy_train)
        st.write("Explicit Accuracy on the test set:", accuracy_test)

        # Confusion matrix
        conf_matrix = (confusion_matrix(y_true=y_test, y_pred=y_pred))

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

    if n_classes == 2:
        y_pred_proba = model.predict_proba(X_test_pca)[:, 1]

    else:
        y_pred_proba = model.predict_proba(X_test_pca)
        

    pca_options = [f"PC{i+1}" for i in range(X_train_resampled.shape[1])]
    st.session_state.model = model

    if Classification_type == "Random Forest Classifier":
        st.write("")

        st.write("Confusion Matrix:")
        st.table(conf_matrix)

        st.write("Classification Report:")
        st.table(report_df)
        
        # Create a DataFrame of feature importances
        pca_importances_df = pd.DataFrame({
            "PCA": [f"PC{i+1}" for i in range(X_train_resampled.shape[1])],
            "Importance": pca_importances
        }).sort_values(by="Importance", ascending=False)

        st.write("The feature importances:")
        st.table(pca_importances_df.head(20))  # Show top 20 features
        # Visualize the top features in a bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(pca_importances_df['PCA'][:20], pca_importances_df['Importance'][:20])  # top 20 features
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importances')
        st.pyplot(plt)

        st.write("Choose two PCA's you want to plot against eachother:")
        # Dropdown for PCA selection
        pca_x = st.selectbox("Select the PCA for the X-axis:", pca_options)
        # Exclude the selected PCA for X-axis from the options for Y-axis
        available_pca_y_options = [pca for pca in pca_options if pca != pca_x]

        # Automatically assign the first available option for Y-axis
        pca_y = st.selectbox(
            "Select the PCA for the Y-axis:",
            available_pca_y_options,
            index=0  # Default to the first available option
        )

        # Extract column indices
        pca_x_index = int(pca_x[2:]) - 1
        pca_y_index = int(pca_y[2:]) - 1

        # Merge PCA data with illness status for plotting
        pca_df = pd.DataFrame(X_train_resampled, columns=pca_options)
        pca_df['Illness'] = y_train_resampled

        # Plotting with Plotly
        fig = px.scatter(
            pca_df,
            x=pca_x,
            y=pca_y,
            color=pca_df['Illness'].map(lambda x: labels[x]),
            title=f"Scatter Plot of {pca_x} vs {pca_y}",
            labels={"color": "Illness Type"},
            color_discrete_sequence=px.colors.qualitative.Set1
        )

        # Show legend and plot
        fig.update_layout(
            legend_title="Illness Types",
            legend=dict(itemsizing="constant"),
            title_x=0.5
        )
        st.plotly_chart(fig)

        st.write("Visualizing an Individual Decision Tree:")

        # Select one tree from the Random Forest
        tree = model.estimators_[0]

        # Plot the tree
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(tree, filled=True, feature_names=[f"PC{i+1}" for i in range(X_train_resampled.shape[1])], 
                  class_names=labels, rounded=True, ax=ax)
        ax.set_title("Decision Tree Visualization (from Random Forest)")
        st.write("Only usefull when there only a few features.")
        st.pyplot(fig)


    elif Classification_type == "Logistic Regression (L1)":
        st.write("")

        st.write("Confusion Matrix:")
        st.table(conf_matrix)

        st.write("Classification Report:")
        st.table(report_df)

        st.write("Coefficients Visualization:")
        coef = model.coef_[0]
        # coef_length = len(coef)
        # aligned_gene_names = [gene_names[i] for i in selected_scores_indices[:coef_length]]
        # st.write(coef)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(X_train_resampled.shape[1]), coef)
        ax.set_xticks(range(X_train_resampled.shape[1]))
        ax.set_xticklabels([f"PC{i+1}" for i in range(X_train_resampled.shape[1])], rotation=90)
        ax.set_title("Logistic Regression (L1) Coefficients")
        ax.set_xlabel("PCA's")
        ax.set_ylabel("Coefficient Value")
        st.pyplot(fig)

        pca_component = st.selectbox("Select PCA component to visualize:", [f"PC{i+1}" for i in range(X_train_resampled.shape[1])])
        # Get the index of the selected PCA component (e.g., PC1 -> index 0)
        pca_index = int(pca_component[2:]) - 1  # e.g., "PC1" -> 0, "PC2" -> 1, etc.
        
        # Extract the loadings (coefficients) for the selected PCA
        pca_component_loadings = pca_components[pca_index]    

        # Create a DataFrame of the loadings (coefficients) of genes for the selected PCA component
        gene_contributions = pd.DataFrame(pca_component_loadings, index=gene_names, columns=['Contribution'])
        # Plot the bar chart with Plotly
        fig = px.bar(
            gene_contributions,
            x=gene_contributions.index,
            y='Contribution',
            title=f"Gene Contributions to {pca_component}",
            labels={'x': 'Genes', 'Contribution': 'Coefficient Value'},
            hover_name=gene_contributions.index,  # Show gene name on hover
            template='plotly_white',
        )
        
        # Update the layout
        fig.update_layout(
            xaxis_title="Genes",  # Set the x-axis title
            yaxis_title="Contribution Value",  # Set the y-axis title
            xaxis_tickangle=-45,  # Rotate x-axis labels if needed
            xaxis=dict(tickmode='array', tickvals=[], ticktext=[]),  # Remove x-axis tick labels
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)


        # Top 20 genes and there contribution plote
        # Assuming pca is your trained PCA object
        gene_contributions = np.dot(model.coef_, pca_components)    

        # Visualize the top gene contributions
        top_genes_indices = np.argsort(np.abs(gene_contributions[0]))[::-1][:20]  # Top 20 genes
        top_gene_names = [gene_names[i] for i in top_genes_indices]
        top_gene_values = gene_contributions[0][top_genes_indices]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(top_gene_names, top_gene_values)
        ax.set_title("Top Gene Contributions After PCA")
        ax.set_xlabel("Gene Names")
        ax.set_ylabel("Contribution Value")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        #gene_contributions = np.dot(pca_components.T, coef)
        #fig, ax = plt.subplots(figsize=(10, 6))
        #ax.bar(range(len(gene_contributions)), gene_contributions)
        #ax.set_xticks(range(len(gene_contributions)))
        #ax.set_xticklabels(gene_names, rotation=90)  # Use actual gene names
        #ax.set_title("Logistic Regression Coefficients (Mapped to Genes)")
        #ax.set_xlabel("Gene Names")
        #ax.set_ylabel("Coefficient Value")
        #st.pyplot(fig)

    elif Classification_type == "XGBoost":
        st.write ("")
        feature_importances = st.session_state.feature_importances
        st.write("Confusion Matrix:")
        st.table(conf_matrix)

        st.write("Classification Report:")
        st.table(report_df)    
        # Feature Importance Plot
        #st.write("Feature Importance Plot:")
        ## Create the plot
        #fig, ax = plt.subplots(figsize=(10, 8))
        #xgb.plot_importance(model, importance_type='gain', max_num_features=20, ax=ax)  # Display top 20 features and 'gain': This calculates feature importance based on the average improvement in loss when a feature is used in splits across all trees. It's a measure of how much the feature contributes to the model's performance.
        #ax.set_title('Feature Importance - XGBoost')
        #
        ## Display the plot in Streamlit
        #st.pyplot(fig)

        # Get PCA components and explained variance ratio
        loadings = pd.DataFrame(
            np.abs(pca_components),
            columns=gene_names
        )

        # Calculate feature importance by summing absolute loadings across components
        feature_importance = loadings.sum(axis=0).sort_values(ascending=False)

        # Plot the top 20 features
        st.write("Feature Importance - PCA")
        fig, ax = plt.subplots(figsize=(10, 8))
        feature_importance.head(20).plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title("Top 20 Feature Importances (PCA)")
        ax.set_ylabel("Cumulative Absolute Loadings")
        ax.set_xlabel("Features")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Display the plot
        st.pyplot(fig)

        # Compute learning curve
        train_sizes, train_scores, test_scores = learning_curve(model, X_train_resampled, y_train_resampled, cv=5)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
        ax.plot(train_sizes, test_scores.mean(axis=1), label='Cross-validation score')
        ax.set_xlabel('Number of training samples')
        ax.set_ylabel('Accuracy')
        ax.set_title('Learning Curve')
        ax.legend(loc='best')

        # Display plot in Streamlit
        st.pyplot(fig)


        explainer = shap.Explainer(model, X_train_resampled)
        shap_values = explainer.shap_values(X_test_pca)  # Use X_test_pca if PCA was applied before the model
        plt.clf() 
        shap.summary_plot(shap_values, X_test_pca, show=False)  # Same for the summary plot, use X_test_pca
        st.write("SHAP (SHapley Additive exPlanation) values:")
        st.pyplot(plt)

        # 2. SHAP Dependence Plot
        # Select a feature (e.g., a PCA component) to visualize
        pca_component = st.selectbox("Select PCA component to visualize:", [f"PC{i+1}" for i in range(X_train_resampled.shape[1])])
        pca_index = int(pca_component[2:]) - 1  # e.g., "PC1" -> 0, "PC2" -> 1, etc.

        st.write(f"SHAP Dependence Plot for {pca_component}:")
        shap.dependence_plot(pca_index, shap_values, X_test_pca, feature_names=[f"PC{i+1}" for i in range(X_test_pca.shape[1])])
        st.pyplot(plt)

        # 3. SHAP Force Plot for a sample (optional)
        # Select a sample (e.g., the first sample) to visualize
        st.write("SHAP Force Plot for the first sample:")
        shap.force_plot(shap_values[0].values, shap_values[0].base_values, shap_values[0].data, show=False)
        st.pyplot(plt)

    elif Classification_type == "One Vs Rest Classifier":
        st.write("Processing the data with the One VS Rest Classifier:")

        st.write("Confusion Matrix:")
        st.table(conf_matrix)

        st.write("Classification Report:")
        st.table(report_df)

        y_test_binarized = label_binarize(y_test, classes=classes)
        fig, ax = plt.subplots()        

        if n_classes == 2:
            precision, recall, _ = precision_recall_curve(y_test_binarized, y_pred_proba)
            ax.plot(recall, precision, lw=2, label="Positive Class")

        else:
            for i, class_label in enumerate(classes):
                precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                ax.plot(recall, precision, lw=2, label=f"Class {class_label}")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        # ax.set_show()
        st.pyplot(fig)

    st.write("ROC curve and AUC:")
    if n_classes == 2:
        # Predictions and Probabilities for ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        # Plot ROC Curve
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')


    else:
        # Initialize plot
        fig, ax = plt.subplots()
        # Identify unique classes
        classes = sorted(set(y_test))
        n_classes = len(classes)

        # Binarize the true labels for multiclass ROC
        y_test_binarized = label_binarize(y_test, classes=classes)
        # Multiclass handling
        if n_classes > 2:
            # Compute ROC curve and AUC for each class
            for i, class_label in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.2f})')
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc='lower right')
    st.pyplot(fig)


else:
    st.error("The data is not ready for machine learning. Please upload and process your data on the previous page.")

if st.button("Next page"):
    switch_page("predicting data")

