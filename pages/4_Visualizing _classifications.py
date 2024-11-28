import streamlit as st
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns   

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve


from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import graphviz

st.set_page_config(page_title="Visualizing The Classifications")
st.markdown("# Visualizing the classification")
st.sidebar.header("Visualizing the classification")
st.write(
    """This page doesn't have to be used, but it can be usefull to use so you have more insight what the classification really did and have a better idea about the data you are using."""
    )

if 'Classification_type' in st.session_state:
    Classification_type = st.session_state.Classification_type
    y_train_resampled = st.session_state.y_train_resampled
    X_train_resampled = st.session_state.X_train_resampled
    classes = st.session_state.classes
    labels = st.session_state.labels
    model = st.session_state.model
    gene_names = st.session_state.gene_names
    labels = st.session_state.labels
    conf_matrix = st.session_state.conf_matrix
    report_df = st.session_state.report_df
    y_test = st.session_state.y_test
    y_pred_proba = st.session_state.y_pred_proba
    pca_components = st.session_state.pca_components
    # selected_scores_indices = st.session_state.selected_scores_indices
    pca_options = [f"PC{i+1}" for i in range(X_train_resampled.shape[1])]

    n_classes = len(classes)
    if n_classes == 2:
        feature_importances = st.session_state.feature_importances
    else:
        pass

    st.write(Classification_type)
    if Classification_type == "Random Forest Classifier":
        st.write("Processing the data with the Random Forest Classifier:")

        st.write("Confusion Matrix:")
        st.table(conf_matrix)

        st.write("Classification Report:")
        st.table(report_df)
        
        # Create a DataFrame of feature importances
        feature_importances_df = pd.DataFrame({
            "Feature": [f"PC{i+1}" for i in range(X_train_resampled.shape[1])],
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)

        # Create a ROC Curve and AUC
        # Predictions and Probabilities for ROC
        #fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        #roc_auc = auc(fpr, tpr)
#
        ## Plot ROC Curve
        #st.write("ROC Curve and AUC:")
        #fig, ax = plt.subplots()
        #ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        #ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        #ax.set_xlim([0.0, 1.0])
        #ax.set_ylim([0.0, 1.05])
        #ax.set_xlabel('False Positive Rate')
        #ax.set_ylabel('True Positive Rate')
        #ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        #ax.legend(loc='lower right')
        #st.pyplot(fig)

        # Display top features in a table
        st.write("The feature importances:")
        st.table(feature_importances_df.head(20))  # Show top 20 features
        # Visualize the top features in a bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances_df['Feature'][:20], feature_importances_df['Importance'][:20])  # top 20 features
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
        st.write("Processing the data with the Logistic Regression:")

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
        ax.set_xlabel("Gene Names")
        ax.set_ylabel("Coefficient Value")
        st.pyplot(fig)

        pca_component = st.selectbox("Select PCA component to visualize:", [f"PC{i+1}" for i in range(X_train_resampled.shape[1])])
        # Get the index of the selected PCA component (e.g., PC1 -> index 0)
        pca_index = int(pca_component[2:]) - 1  # e.g., "PC1" -> 0, "PC2" -> 1, etc.
        
        # Extract the loadings (coefficients) for the selected PCA
        pca_component_loadings = pca_components[pca_index]    

        # Create a DataFrame of the loadings (coefficients) of genes for the selected PCA component
        gene_contributions = pd.DataFrame(pca_component_loadings, index=gene_names, columns=[pca_component])

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the bar chart of gene contributions
        sns.barplot(x=gene_contributions.index, y=gene_contributions[pca_component], ax=ax)
        ax.set_title(f"Gene Contributions to {pca_component}")
        ax.set_xlabel("Genes")
        ax.set_ylabel("Coefficient Value")
        ax.set_xticklabels(gene_contributions.index, rotation=90)

        # Show the plot
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

    elif Classification_type == "One Vs Rest Classifier":
        st.write("Processing the data with the One VS Rest Classifier:")

        st.write("Confusion Matrix:")
        st.table(conf_matrix)

        st.write("Classification Report:")
        st.table(report_df)

        y_test_binarized = label_binarize(y_test, classes=classes)
        fig, ax = plt.subplots()        

        for i, class_label in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
            ax.plot(recall, precision, lw=2, label=f"Class {class_label}")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve for Multiclass Classification")
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
    st.warning("Please run the 'Machine Learning' page first.")
