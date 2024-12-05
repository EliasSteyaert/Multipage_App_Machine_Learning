import streamlit as st

def load_css():
    st.markdown("""
    <style>
    /* Style for the tooltip icon and the question mark emoji */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        color: inherit;  /* Inherit text color from parent */
        font-size: inherit;  /* Inherit font size */
        text-decoration: none;  /* Remove underline */
        padding: 10px;
    }
    
    .emoji {
        font-size: 1.2em;  /* Slightly larger size for the emoji */
        margin-left: 5px;  /* Space between the text and the emoji */
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #f9f9f9;
        color: #333;
        text-align: center;
        border-radius: 5px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
    }

       /* Adjust margin for markdown (the title or explanation above the slider) */
    .stMarkdown {
        margin-bottom: -10px;  /* Adjust the space between the title and slider */
        margin-top: -10px;
    }
    .stSlider {
        margin-top: -40px;  /* Reduce the space above the slider */
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)