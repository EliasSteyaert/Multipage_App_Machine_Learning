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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import plotly.graph_objects as go
from kneed import KneeLocator

import seaborn as sns

from streamlit_extras.switch_page_button import switch_page
from styles import load_css

st.set_page_config(page_title="Preprocessing")
st.markdown("# The data preprocessing part:")
st.sidebar.header("Preprocessing")
st.write(
    """This is the data preprocessing configuration page. Select the options you want to use for preprocessing the data. Modify the options and parameters if needed. If the results aren't to your satisfaction after the model has been created, you can get the accuracy of the testing data to your desired heights with adjusting the parameters."""
    )

# Check if 'data' exists in session_state
if "data" in st.session_state and st.session_state["data"] is not None:
    data = st.session_state["data"]

    load_css()

    st.write("Starting the preprocessing part...")
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
   
    # Tooltip explanation
    tooltip_text = """
    When clicked upon one of the options, underneath shall the explanation be shown.
    """

    # Add a tooltip to the radio button
    st.markdown("""
        <p style="margin-top: 20px; margin-bottom: -35px;">
            <span class="tooltip">
                <span class="emoji">❓</span>
                Do you want to <strong>balance the data</strong> with an oversampling technique?
                <span class="tooltiptext">{}</span>
            </span>
        </p>
        """.format(tooltip_text), unsafe_allow_html=True)

    # Radio button for data balancing
    sample_distribution = st.radio(" ", options=["SMOTE", "ADASYN", "No"])
    if sample_distribution == "SMOTE":
        st.write("The data will be processed with the SMOTE (Synthetic Minority Oversampling Technique) algorithm. This technique generates synthetic samples by interpolating between existing minority class samples. Works well for both binary and multiclass data with moderate imbalance. It is particularly useful when dealing with highly imbalanced data, as it helps to balance the class distribution, which can improve model performance by reducing bias toward the majority class.")
        st.write("")
        st.write("")

    elif sample_distribution == "ADASYN":
        st.write("The data will be processed with the ADASYN (Adaptive Synthetic Sampling) oversampling technique. This technique ocuses on difficult-to-learn samples by generating more synthetic data in sparse areas of the minority class distribution. Useful for both binary and multiclass imbalances. Suitable for highly imbalanced datasets or cases with hard-to-classify samples. Works well for multiclass data, especially when decision boundaries need fine-tuning. Do not use when the data is noisy.")
        st.write("")
        st.write("")
                    
    elif sample_distribution == "No":
        st.write("The data will not be processed with an oversampling technique. Keep in mind the data has to be balanced for the classification step.")
        st.write("")
        st.write("")

    # Storing the data in 'X' and 'y'
    X=data.iloc[:, 1:-1] # Only gene expression columns
    y=data.iloc[:, -1] # Only illness status

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
    st.write("")

    # split data into training and test sets
    X_train,X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.2,random_state=42)
    # random state = 42 : Ensures reproducibility by using a fixed random seed for splitting the data.
    st.write("The data has been successfully split into training (80%) and testing (20%) sets.")
    st.write("")
    

    # Mutual Information Feature Selection with slider
    # st.header("Mutual Information-Based Feature Selection")
    st.markdown("""
    <p style="margin-top: 20px; margin-bottom: -35px;">
        <span class="tooltip"> <span class="emoji">❓</span>Select the <b>number of features</b> you want when performing Feature Selection:
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
    step=1      
    )

    st.write("Performing the feature selection step with cross-validation:")
    MI=mutual_info_classif(X_train,y_train)
    #select top n features. lets say 300.
    #you can modify the value and see how the performance of the model changes
    # n_features=300
    selected_scores_indices=np.argsort(MI)[::-1][0:n_features]
    
    X_train_selected=X_train.iloc[:, selected_scores_indices]
    X_test_selected=X_test.iloc[:, selected_scores_indices]

    # Evaluate with cross-validation
    model = RandomForestClassifier(random_state=42)
    scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='f1_weighted')
    st.write("The feature selection step was successfully performed.")
    st.write("The cross-validation score:")

    # Tooltip text with dynamic content
    tooltip_text = f"""
    High Score (close to 1.0): Indicates that the model is performing well at distinguishing between classes, with both precision and recall being strong.
    Moderate Score (0.6 to 0.8): Suggests the model has room for improvement. This could be due to insufficiently informative features, data imbalance, or model limitations.
    Low Score (below 0.5): Highlights significant issues with the model's ability to classify correctly, possibly due to noisy data, incorrect preprocessing, or insufficient features.
    Adjust the number of features ({n_features}) to improve performance.
    """

    # Title with emoji and tooltip
    title_html = f"""
    <p>
        <span class="tooltip">
            <span class="emoji">❓</span>
            Mean F1 Score with top {n_features} features: {np.mean(scores):.4f}<br>
            <span class="tooltiptext">{tooltip_text}</span>
        </span>
    </p>
    """

    # Display the title in Streamlit with the tooltip functionality
    st.markdown(title_html, unsafe_allow_html=True)
    st.write("")
 
    # scale data between 0 and 1
    min_max_scaler = MinMaxScaler()
    X_train_norm = min_max_scaler.fit_transform(X_train_selected)
    X_test_norm = min_max_scaler.transform(X_test_selected)
    st.session_state.min_max_scaler = min_max_scaler
    st.write("The Normalization step was successful.")

    # Scree Plot for PCA
    pca = PCA()
    pca.fit(X_train_norm)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot for explained variance ratio
    x = np.arange(1, len(explained_variance_ratio) + 1)

    # Find elbow point using custom logic
    elbow_point = np.argmax(np.diff(cumulative_variance_ratio) < 0.01) + 1  # Adding 1 for 1-based indexing
    elbow_cumulative = cumulative_variance_ratio[elbow_point - 1]  # Adjust index for Python 0-based indexing

    # Create Plotly figure
    fig = go.Figure()

    # Add explained variance bars
    fig.add_trace(go.Bar(
        x=x,
        y=explained_variance_ratio,
        name="Explained Variance Ratio",
        marker=dict(color="skyblue"),
        hoverinfo="x+y"
    ))

    # Add cumulative variance line
    fig.add_trace(go.Scatter(
        x=x,
        y=cumulative_variance_ratio,
        mode='lines+markers',
        name="Cumulative Variance",
        line=dict(color="red", width=2),
        marker=dict(size=8),
        hoverinfo="x+y",
        text=[f"PC {i}: {v:.2%}" for i, v in enumerate(cumulative_variance_ratio, start=1)],  # Hover text
    ))

    # Highlight elbow point
    fig.add_trace(go.Scatter(
        x=[elbow_point],
        y=[elbow_cumulative],
        mode='markers',
        name="Elbow Point",
        marker=dict(color="gold", size=12, symbol='star'),
        hoverinfo="x+y",
        text=[f"Elbow at PC {elbow_point}: {elbow_cumulative:.2%} cumulative variance"]
    ))

    # Layout adjustments
    fig.update_layout(
        title={
        'text': "Interactive Scree Plot with Elbow Point",
        'x': 0.5,  # Centers the title
        'xanchor': 'center'  # Ensures the title is centered correctly
    },
        xaxis=dict(
            title="Principal Component",
            tickmode='array',
            tickvals=x if len(x) <= 15 else None,  # Show ticks only if fewer components
            ticktext=x if len(x) <= 15 else None,  # Show tick text only if fewer components
        ),
        yaxis=dict(title="Variance Explained"),
        showlegend=True,
        template="plotly_white",
        height=600
    )

    # Show in Streamlit
    st.plotly_chart(fig)

    
    # User selects number of components for PCA 
    st.markdown("""
    <p style="margin-top: -10px; margin-bottom: -35px;">
        <span class="tooltip"><span class="emoji">❓</span>Select the <b>number of components</b> you want when performing PCA:
            <span class="tooltiptext">The scree plot helps determine how many components to use by showing the variance each one explains. Components beyond the 'elbow point' add little value, so you can exclude them to reduce computation. The default of 8 components is set to balance efficiency with retaining enough variance for meaningful analysis."
            </span>
        </span>
    </p>
    """, unsafe_allow_html=True)
    n_components = st.slider(
        label = "",
        min_value=1,
        max_value=len(explained_variance_ratio),
        value=8  # Default value
    )

    # Principal component analysis steps
    pca = PCA(n_components=n_components)  # Adjust n_components based on explained variance
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    st.session_state.pca = pca
    st.session_state.X_train_pca = X_train_pca

    st.write("The Principal Component Analysis was succcessful.")
    st.write("")

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

    gene_names = X_train.columns[selected_scores_indices]
    st.session_state.gene_names = gene_names
    st.session_state.y_train_resampled = y_train_resampled
    st.session_state.X_train_resampled = X_train_resampled
    st.session_state.X_train_pca = X_train_pca
    st.session_state.y_test = y_test
    st.session_state.label_encoder = label_encoder
    st.session_state.pca_components = pca.components_
    st.session_state.n_components = n_components
    st.session_state.X_test_pca = X_test_pca
    st.session_state.X_train_selected = X_train_selected 

if st.button("Next page"):
    switch_page("machine learning")
