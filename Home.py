import streamlit as st

st.set_page_config(
    page_title="Multipage Machine Learning"
)

st.sidebar.header("Home")
st.write("# Machine Learning on Cancer Data ")

st.sidebar.success("Select which step you want to perform.")

st.markdown(
    """
    Welcome to the machine learning script that predicts a patient's cancer status based on their gene expression data.  
    Start by uploading the training data (Page 1) to create the machine learning model. You can adjust various steps of the process to customize it to your needs (Page 2).  
    Finally, upload your patient's data to predict their illness status and view the results (Page 3).
    """
)