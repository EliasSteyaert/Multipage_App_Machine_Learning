import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from streamlit_extras.switch_page_button import switch_page

from styles import load_css

st.set_page_config(page_title="Volcano plot")
st.markdown("# Create a volcano plot.")
st.sidebar.header("Creating a volcano plot")
st.write(
    """You don't have to use this page. This page can only be used when you have two files and the table with the gene expressions also includes a logFc and a P-value column (or multiple).
    Make sure that the contrast of both the logFc and the P-value are the same, otherwise the volcanoplot will not be as good as it should be."""
    )


if "data" in st.session_state and st.session_state["data"] is not None:
    data = st.session_state["data"]
    topTables = st.session_state["topTables"]
    gene_names = st.session_state["gene_names"]

    load_css()

    log_fc_column = st.selectbox("Select the column with the desired **logFc data**:", topTables.columns)
    p_val_column = st.selectbox("Select the column with the desired **P-value data**:", topTables.columns)
    
    try:
        # Create a DataFrame with the selected columns and gene names
        vulcano_plot_data = pd.DataFrame({
            "Gene Names": topTables[gene_names],
            "Log Fold Change": topTables[log_fc_column],
            "P-Value": topTables[p_val_column]
        })
    except KeyError as e:
        st.warning(f"KeyError: {e}. One or more specified columns are missing from the data. Skipping volcano plot data creation.")
        vulcano_plot_data = None  # Initialize as None or take other actions if needed.
    
    vulcano_plot_data["Minus log10 P-Value"] = -np.log10(vulcano_plot_data["P-Value"])

    st.markdown("""
    <p>
        <span class="tooltip"><span class="emoji">❓</span>Choose the threshold that you want for the <b>logFC</b>: 
            <span class="tooltiptext">LogFC (log fold change) measures the magnitude of change in gene expression. Adjust this to filter genes with significant changes.<br>
                |logFC| ≥ 1 is commonly used, as it indicates a twofold change in expression between conditions (e.g., healthy vs. diseased).<br>
                If the dataset is noisy or has subtle changes, you might lower the threshold to |logFC| ≥ 0.5.</span>
        </span>
    </p>
    """, unsafe_allow_html=True)
    # Define thresholds
    log_fc_threshold = st.slider(
    label="",
    min_value=0.0,
    max_value=2.5,
    value=1.0,
    step=0.1
    )
    
    st.markdown("""
<p>
    <span class="tooltip"><span class="emoji">❓</span>Choose the threshold that you want for the <b>-log10(P-value)</b>: 
        <span class="tooltiptext">P-value indicates statistical significance. Lower values mean stronger evidence against the null hypothesis.<br>
                If you want to be conservative and reduce the likelihood of false positives, a higher threshold is better. <br>
                If you are exploring potential trends in the data and want to include genes with even marginal significance, you can use a lower threshold, such as 1.3, which corresponds to a P-value of ≤ 0.05. (2 corresponds to a P-value of ≤ 0.01)
        </span>
    </span>
</p>
""", unsafe_allow_html=True)
    p_val_threshold = st.slider(
    label="",
    min_value=0.00,
    max_value=2.5,
    value=1.3,
    step=0.1
    )

    # Create a new column to indicate whether genes are inside the thresholds
    vulcano_plot_data['outside_threshold'] = (
        (vulcano_plot_data["Log Fold Change"].abs() > log_fc_threshold) & 
        (vulcano_plot_data["Minus log10 P-Value"] > p_val_threshold)
    )
    genes_outside_threshold = vulcano_plot_data[vulcano_plot_data['outside_threshold']]['Gene Names'].tolist()
    amount_outside_genes = len(genes_outside_threshold)

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

    # Ensure genes_inside_threshold is also stripped so that a simple space can not create confusion and so an error for the algorithm, but I don't think this is needed anymore
    genes_outside_threshold = [gene.strip() for gene in genes_outside_threshold]

    # Radio button for single selection between two options
    st.write("Do you want to **drop the genes** that fall out of the choosen thresholds (",amount_outside_genes,"amount of genes would remain)?")
    genes_usage = st.radio("Select your desired option:", options=["no", "yes"])
    if genes_usage == "yes":
        st.write("Processing data where the insignifcant genes gets dropped:")  
        # Identify the first and last columns
        first_column = data.columns[0]
        last_column = data.columns[-1]

        # Retain the first and last columns and the ones in 'genes_inside_threshold'
        columns_to_keep = [first_column] + \
                          [col for col in data.columns if col in genes_outside_threshold] + \
                          [last_column]

        # Filter the DataFrame to retain only the desired columns
        data = data.loc[:, data.columns.isin(columns_to_keep)]

        # Check the updated DataFrame
        st.write(data)

        st.session_state.data = data
        st.write("The shape of the data after dropping the genes inside the thresholds:", data.shape)

    elif genes_usage == "no":
         st.write("The data remains unchanged in this step.")
    
    column_names_list = data.columns[1:].tolist()
    column_to_check = "ensembl_gene_id"
    filtered_topTables = topTables[topTables[column_to_check].isin(column_names_list)]

else:
    st.error("Please complete the 'uploading data' page first.")

if st.button("Next page"):
    switch_page("preprocessing")


