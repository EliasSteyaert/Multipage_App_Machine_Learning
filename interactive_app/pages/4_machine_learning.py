import streamlit as st
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd

from sklearn.preprocessing import LabelEncoder
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
    label_encoder = st.session_state.label_encoder
    y_test = st.session_state.y_test
    pca = st.session_state.pca
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
        original_importances = np.abs(np.dot(pca_components.T, pca_importances))


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

        # Assuming original_importances contains the mapped importance values from PCA to original features
        top_genes_indices = np.argsort(np.abs(original_importances))[::-1][:20]  # Top 20 genes
        top_gene_names = [gene_names[i] for i in top_genes_indices]  # Replace 'gene_names' with your actual gene list
        top_gene_values = original_importances[top_genes_indices]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(top_gene_names, top_gene_values)
        ax.set_title("Top 20 Gene Importances")
        ax.set_xlabel("Gene Names")
        ax.set_ylabel("Feature Importance (Impurity Reduction)")
        ax.tick_params(axis='x', rotation=90)  # Ensure vertical labels
        plt.xticks(rotation=90)  # This ensures proper vertical alignment as a fallback
        st.pyplot(fig)

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

        # Get explained variance ratio (same for binary and multiclass)
        explained_variance = pca.explained_variance_ratio_

        # Plotting the explained variance ratio for each PCA component
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, len(explained_variance) + 1), explained_variance, color='skyblue')

        # Add labels and title
        ax.set_xticks(range(1, len(explained_variance) + 1))
        ax.set_xticklabels([f'PC{i}' for i in range(1, len(explained_variance) + 1)], rotation=90)
        ax.set_title('PCA Components Explained Variance')
        ax.set_xlabel('PCA Component')
        ax.set_ylabel('Explained Variance Ratio')

        # Display the plot in Streamlit
        st.pyplot(fig)

        st.write("Coefficients Visualization:")
        if n_classes == 2:
            # Get the coefficients in PCA-transformed space
            coefficients = model.coef_.flatten() # Shape: (n_classes, n_components)
            original_importances = np.dot(coefficients, pca_components)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(coefficients)), coefficients)  # Use 1D array for bar heights
            ax.set_xticks(range(len(coefficients)))
            ax.set_xticklabels([f"PC{i+1}" for i in range(len(coefficients))], rotation=90)
            ax.set_title("Logistic Regression (L1) Coefficients")
            ax.set_xlabel("PCA components")
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

            top_genes_indices = np.argsort(np.abs(original_importances))[::-1][:20]  # Top 20 genes
            top_gene_names = [gene_names[i] for i in top_genes_indices]
            top_gene_values = original_importances[top_genes_indices]
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(top_gene_names, top_gene_values)
            ax.set_title("Top Gene Contributions After PCA")
            ax.set_xlabel("Gene Names")
            ax.set_ylabel("Contribution Value")
            plt.xticks(rotation=90)
            st.pyplot(fig)
            
        else:
            coefficients = model.coef_
            original_importances = np.dot(coefficients, pca_components)  # Multiply transposed coefficients with PCA components

            fig, ax = plt.subplots(figsize=(10, 6))
            
            for i in range(coefficients.shape[0]):  # Loop through each class
                ax.bar(range(coefficients.shape[1]), coefficients[i, :], label=f"Class {i+1}")

            ax.set_xticks(range(coefficients.shape[1])) 
            ax.set_xticklabels([f"PC{i+1}" for i in range(coefficients.shape[1])], rotation=90)
            ax.set_title("Logistic Regression (L1) Coefficients by Class")
            ax.set_xlabel("PCA Components")
            ax.set_ylabel("Coefficient Value")
            ax.legend()
            st.pyplot(fig)


            # Compute combined importance for all classes
            absolute_importances = np.abs(original_importances)  # Absolute values of contributions
            combined_importance = np.sum(absolute_importances, axis=0)  # Sum across all classes
            
            # Add "All Classes Combined" to the class options
            class_options = [f"Class {i+1}" for i in range(coefficients.shape[0])]
            class_options.append("All Classes Combined")  # Add the combined option
            
            # Create the selectbox
            selected_option = st.selectbox("Choose a class to investigate or view combined contributions:", class_options)
            
            # Determine the behavior based on the selection
            if selected_option == "All Classes Combined":
                # Top 20 genes for all classes combined
                top_genes_indices = np.argsort(combined_importance)[::-1][:20]  # Indices of top 20 genes
                top_gene_names = [gene_names[i] for i in top_genes_indices]  # Gene names
                top_gene_values = combined_importance[top_genes_indices]  # Contribution values
            
                # Plot the combined contributions
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(top_gene_names, top_gene_values)
                ax.set_title("Top 20 Gene Contributions Across All Classes")
                ax.set_xlabel("Gene Names")
                ax.set_ylabel("Combined Contribution Value")
                plt.xticks(rotation=90)
                st.pyplot(fig)
            
            else:
                # Extract the class index
                class_index = int(selected_option.split(" ")[1]) - 1  # "Class 1" -> index 0
            
                # Top 20 genes for the selected class
                top_genes_indices = np.argsort(np.abs(original_importances[class_index]))[::-1][:20]
                top_gene_names = [gene_names[i] for i in top_genes_indices]
                top_gene_values = original_importances[class_index][top_genes_indices]
            
                # Plot the contributions for the selected class
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(top_gene_names, top_gene_values)
                ax.set_title(f"Top 20 Gene Contributions for {selected_option}")
                ax.set_xlabel("Gene Names")
                ax.set_ylabel("Contribution Value")
                plt.xticks(rotation=90)
                st.pyplot(fig)

    elif Classification_type == "XGBoost":
        st.write ("")
        st.write("Confusion Matrix:")
        st.table(conf_matrix)

        st.write("Classification Report:")
        st.table(report_df)    

        # Get explained variance ratio (same for binary and multiclass)
        explained_variance = pca.explained_variance_ratio_

        # Plotting the explained variance ratio for each PCA component
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, len(explained_variance) + 1), explained_variance, color='skyblue')

        # Add labels and title
        ax.set_xticks(range(1, len(explained_variance) + 1))
        ax.set_xticklabels([f'PC{i}' for i in range(1, len(explained_variance) + 1)], rotation=90)
        ax.set_title('PCA Components Explained Variance')
        ax.set_xlabel('PCA Component')
        ax.set_ylabel('Explained Variance Ratio')

        # Display the plot in Streamlit
        st.pyplot(fig)

         # Map feature importances back to original features
        pca_importances = model.feature_importances_
        original_importances = np.dot(pca_components.T, pca_importances)

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

    elif Classification_type == "One Vs Rest Classifier":
        st.write("Processing the data with the One VS Rest Classifier:")

        st.write("Confusion Matrix:")
        st.table(conf_matrix)

        st.write("Classification Report:")
        st.table(report_df)

        # Get explained variance ratio (same for binary and multiclass)
        explained_variance = pca.explained_variance_ratio_

        # Plotting the explained variance ratio for each PCA component
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, len(explained_variance) + 1), explained_variance, color='skyblue')

        # Add labels and title
        ax.set_xticks(range(1, len(explained_variance) + 1))
        ax.set_xticklabels([f'PC{i}' for i in range(1, len(explained_variance) + 1)], rotation=90)
        ax.set_title('PCA Components Explained Variance')
        ax.set_xlabel('PCA Component')
        ax.set_ylabel('Explained Variance Ratio')

        # Display the plot in Streamlit
        st.pyplot(fig)

        if n_classes == 2:
            # Get the feature importances for the Random Forest model
            feature_importances = model.estimators_[0].feature_importances_  # Use the first estimator for a binary classifier

            # This will give us the original feature importances in the PCA space
            original_feature_importances = np.dot(pca_components.T, feature_importances)

            # Create a DataFrame with the original feature names and importances
            importance_df = pd.DataFrame({
                "Feature": gene_names,  # List of original feature names (genes)
                "Importance": np.abs(original_feature_importances)
            })

            # Sort by importance and get the top 20 features
            importance_df_sorted = importance_df.sort_values(by="Importance", ascending=False).head(20)


            # Plot the top 20 most important features (vertical bars)
            fig, ax = plt.subplots(figsize=(10, 6))
            importance_df_sorted.plot(kind='bar', x='Feature', y='Importance', legend=False, ax=ax)

            # Set plot title and labels
            plt.title("Top 20 Important Genes")
            plt.ylabel("Importance")
            plt.xlabel("Feature")

            # Rotate the x-axis labels to be vertical
            plt.xticks(rotation=90)

            # Display the plot in Streamlit
            st.pyplot(fig)
        else:
            # Create an empty dictionary to store importances for each class
            importances_dict = {}
            # Create selectbox to choose class or combined importance
            class_options = list(label_encoder.classes_) + ["All Classes Combined"]
            selected_option = st.selectbox(
                "Choose a class to investigate or view combined contributions:", class_options
            )
            if selected_option == "All Classes Combined":
                # Loop over all the classes (estimators) and calculate the feature importances
                for idx, estimator in enumerate(model.estimators_):
                    # Get the feature importances for this class (shape: 8)
                    class_importances = estimator.feature_importances_

                    # Project the importances back to the original features
                    original_feature_importances = np.dot(pca_components.T, class_importances)  # Shape: (300,)

                    # Store the importances in the dictionary
                    importances_dict[f'Class_{idx}'] = original_feature_importances
                # Calculate combined feature importance by averaging across all classes
                combined_importance = np.mean(np.array(list(importances_dict.values())), axis=0)

                # Create a DataFrame to hold the feature importances and their corresponding original feature names
                importance_df = pd.DataFrame({
                    "Feature": gene_names,  # Original feature names (length 300)
                    "Importance": combined_importance  # Combined feature importance for all classes
                })

                # Sort the DataFrame by importance and select the top 20 features
                importance_df_sorted = importance_df.sort_values(by="Importance", ascending=False).head(20)

                # Visualize the top 20 features using a vertical bar plot
                fig, ax = plt.subplots(figsize=(10, 6))
                importance_df_sorted.plot(kind='bar', x='Feature', y='Importance', legend=False, ax=ax)

                # Rotate the x-axis labels vertically for better readability
                plt.xticks(rotation=90)

                # Add titles and labels
                plt.title(f"Top 20 Important Features for All Classes Combined")
                plt.xlabel("Feature")
                plt.ylabel("Importance")

                # Display the plot
                st.pyplot(fig)

            else:
                # Convert the selected class name to the appropriate encoded value
                selected_class_encoded = label_encoder.transform([selected_option])[0]

                # Loop over all the classes (estimators) and calculate the feature importances
                for idx, estimator in enumerate(model.estimators_):
                    # Get the feature importances for this class (shape: 8)
                    if idx == selected_class_encoded:
                        # Get the feature importances for this class (shape: 8)
                        class_importances = estimator.feature_importances_

                        # Project the importances back to the original features
                        original_feature_importances = np.dot(pca_components.T, class_importances)  # Shape: (300,)

                        # Store the importances in the dictionary for the selected class
                        importances_dict[f'Class_{selected_class_encoded}'] = original_feature_importances

                        # Create a DataFrame to hold the feature importances and their corresponding original feature names
                        importance_df = pd.DataFrame({
                            "Feature": gene_names,  # Original feature names
                            "Importance": original_feature_importances  # Feature importances for the selected class
                        })

                        # Sort the DataFrame by importance and select the top 20 features
                        importance_df_sorted = importance_df.sort_values(by="Importance", ascending=False).head(20)

                        # Visualize the top 20 features using a vertical bar plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        importance_df_sorted.plot(kind='bar', x='Feature', y='Importance', legend=False, ax=ax)
    
                        # Rotate the x-axis labels vertically for better readability
                        plt.xticks(rotation=90)

                        # Add titles and labels
                        plt.title(f"Top 20 Important Features for {selected_option}")
                        plt.xlabel("Feature")
                        plt.ylabel("Importance")

                        # Display the plot
                        st.pyplot(fig)
                        break  # Exit the loop once the correct class is found and processed


    if n_classes == 2:
        pass
    else: 
        # Display class-label mapping table
        class_label_df = pd.DataFrame({
            'Class Label': range(len(labels)),
            'Class Name': labels
        })
        st.write("Class-Label Mapping Table:")
        st.write(class_label_df)

    # Precision-Recall Curve
    st.write("Precision-Recall Curve:")
    fig, ax = plt.subplots()

    # Binarize the true labels for Precision-Recall
    y_test_binarized = label_binarize(y_test, classes=classes)

    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_test_binarized, y_pred_proba)
        ax.plot(recall, precision, lw=2, label="Positive Class")
    else:
        for i, class_label in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            ax.plot(recall, precision, lw=2, label=f"{class_label}")  # Use class names here

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    st.pyplot(fig)

    # ROC Curve and AUC
    st.write("ROC curve and AUC:")
    fig, ax = plt.subplots()

    if n_classes == 2:
        # Binary Classification
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
    else:
        # Multiclass Classification
        y_test_binarized = label_binarize(y_test, classes=classes)  # Binarize the labels for multiclass

        for i, class_label in enumerate(classes):  # Iterate over class names
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')  # Use class names here

        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
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

