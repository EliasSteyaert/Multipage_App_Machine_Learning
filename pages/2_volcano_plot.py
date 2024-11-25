import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(page_title="Volcano plot")
st.markdown("# Create a volcano plot.")
st.sidebar.header("Creating a volcano plot")
st.write(
    """You don't have to use this page. This page can only be used when you have two files and the table with the gene expressions also includes a logFc and a P-value file (or multiple).
    Make sure that the contrast of both the logFc and the P-value are the same, otherwise the volcanoplot will not be as good as it should be."""
    )


if "data" in st.session_state and st.session_state["data"] is not None:
    data = st.session_state["data"]
    topTables = st.session_state["topTables"]
    gene_names = st.session_state["gene_names"]

    log_fc_column = st.selectbox("Select the column with the desired logFc data:", topTables.columns)
    p_val_column = st.selectbox("Select the column with the desired P-value data:", topTables.columns)
    
    # Create a DataFrame with the selected columns and gene names
    vulcano_plot_data = pd.DataFrame({
        "Gene Names": topTables[gene_names],
        "Log Fold Change": topTables[log_fc_column],
        "P-Value": topTables[p_val_column]
    })
    
    vulcano_plot_data["Minus log10 P-Value"] = -np.log10(vulcano_plot_data["P-Value"])

    # Define thresholds
    log_fc_threshold = st.slider(
    "Choose the threshold that you want for the logFC:",
    min_value=0.0,
    max_value=2.5,
    value=1.0,
    step=0.1
    )
    
    p_val_threshold = st.slider(
    "Choose the threshold that you want for the P-value:",
    min_value=0.0,
    max_value=5.0,
    value=0.5,
    step=0.1
    )

    # Create a new column to indicate whether genes are inside the thresholds
    vulcano_plot_data['inside_threshold'] = (
        (vulcano_plot_data["Log Fold Change"].abs() < log_fc_threshold) & 
        (vulcano_plot_data["Minus log10 P-Value"] < p_val_threshold)
    )

    genes_inside_threshold = vulcano_plot_data[vulcano_plot_data['inside_threshold']]['Gene Names'].tolist()

    ## Display the resulting DataFrame
    #st.write("Combined Data for Volcano Plot:")
    #st.dataframe(vulcano_plot_data)

    # Volcano plot
    fig = px.scatter(
        vulcano_plot_data, 
        x="Log Fold Change", 
        y="Minus log10 P-Value", 
        title="Volcano Plot",
        labels={"Log Fold Change": "Log Fold Change", "Minus log10 P-Value": "-log10(P-value)"},
        hover_data=["Gene Names"]
    )

    # Add lines to the plot for the thresholds
    fig.add_hline(y=p_val_threshold, line_dash="dash", line_color="red", annotation_text="P-value threshold (0.5)")
    fig.add_vline(x=log_fc_threshold, line_dash="dash", line_color="blue", annotation_text="LogFC threshold (1)")
    fig.add_vline(x=-log_fc_threshold, line_dash="dash", line_color="blue")
    
    st.plotly_chart(fig)

    st.write("The shape of the data as of now:", data.shape)

    # Clean column names in the DataFrame
    data.columns = data.columns.str.strip()  # Remove leading/trailing spaces   

    # Ensure genes_inside_threshold is also stripped
    genes_inside_threshold = [gene.strip() for gene in genes_inside_threshold]

    # Radio button for single selection between two options
    st.write("Do you want to drop the genes that don't show a significant expression?")
    genes_usage = st.radio("Select your desired option:", options=["no", "yes"])
    if genes_usage == "yes":
        st.write("Processing data where the insignifcant genes gets dropped:")  
        data = data.drop(genes_inside_threshold, axis=1, errors='ignore')
        st.session_state.data = data
        st.write("The shape of the data after dropping the genes inside the thresholds:", data.shape)

    elif genes_usage == "no":
         st.write("The data remains unchanged in this step.")    

else:
    st.error("Please complete the 'uploading data' page first.")

