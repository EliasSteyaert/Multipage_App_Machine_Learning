import streamlit as st
import styles
from streamlit_extras.switch_page_button import switch_page



st.set_page_config(
    page_title="Multipage Machine Learning"
)

st.sidebar.header("Home")
st.write("# Machine Learning on Cancer Data ")

st.sidebar.success("Select which step you want to perform.")

st.markdown(
    """
    Welcome to the machine learning multipage app that predicts a patient's cancer status based on their gene expression data. This app is made for doctors/data analysists that want to predict the illness status of patients based on previously found data on the same type of illness.

    Start by uploading the training data (uploading data) to begin the creation of the machine learning model. After this you follow the pages on the multipage app. There is an easy button at the end of each page to get you to the next one, or you can navigate the tabs through clicking on one of the pages in the sidebar. The third page (volcano plot) can be skipped if you don’t want to perform that step or the data didn’t have a differential gene expression step earlier.

    Throughout the process, there will be some “❓” along the way. When you hover the mouse upon that sign or the  corresponding text, a tooltip will be shown.
    
    On the second page (uploading data), the app will ask a bit of information of your dataset so it can automatically perform the machine learning. Make sure that you enter the right columns and that the right checkboxes are selected.
    
    If your dataset is in a different shape than what I anticipated, you might have to transpose your dataset first.
    
    Starting from the fourth page (preprocessing data), you are able to choose your own settings on some parts of the machine learning. Depending on the data that is being used (and for example, how it is distributed), you should adjust the settings to your likings and create the highest accuracy score you should be able to get (without over-/underfitting). Feel free to toy a bit with it!
    
    I hope this app simplifies the process of applying machine learning to gene expression data and helps you make valuable predictions for cancer research or clinical applications.
    
    Greetings,
    
    Elias    
    """
)

if st.button("Next page"):
    switch_page("uploading data")
