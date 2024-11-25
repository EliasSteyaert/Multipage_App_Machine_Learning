import streamlit as st
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

st.set_page_config(page_title="Visualizing The Classifications")
st.markdown("# Visualizing the classification")
st.sidebar.header("Visualizing the classification")
st.write(
    """This page doesn't have to be used, but it can be usefull to use so you have more insight what the classification really did and have a better idea about the data you are using."""
    )
st.write("test1")
if 'Classification_type' in st.session_state:
    st.write("test2")
    Classification_type = st.session_state.Classification_type
    X_train_pca = st.session_state.X_train_pca
    feature_importances = st.session_state.feature_importances
    pca_options = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
    st.write("test3")
    if Classification_type == "Random Forest Classification":
        st.write("Processing the data with the Random Forest Classifier:")
        # Create a DataFrame of feature importances
        feature_importances_df = pd.DataFrame({
            "Feature": [f"PC{i+1}" for i in range(X_train_pca.shape[1])],
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)

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
        pca_y = st.selectbox("Select the PCA for the Y-axis:", pca_options)

        # Extract column indices
        pca_x_index = int(pca_x[2:]) - 1
        pca_y_index = int(pca_y[2:]) - 1

        # Scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_pca[:, pca_x_index], X_train_pca[:, pca_y_index], alpha=0.7)
        plt.xlabel(pca_x)
        plt.ylabel(pca_y)
        plt.title(f"Scatter Plot of {pca_x} vs {pca_y}")
        st.pyplot(plt)
    
    elif Classification_type == "Logistic Regression (L1)":
        st.write("Processing the data with the Logistic Regression:")

    elif Classification_type == "One Vs Rest Classifier":
        st.write("Processing the data with the One VS Rest Classifier:")

else:
    st.warning("Please run the 'Machine Learning' page first.")
