import streamlit as st
import re

def next_page():
    """Increases the page number in session state."""
    if "page" not in st.session_state:
        st.session_state.page = 1  # Initialize page state if not already set
    st.session_state.page += 1

def previous_page():
    """Decreases the page number in session state."""
    if "page" in st.session_state and st.session_state.page > 1:
        st.session_state.page -= 1

def show_next_button():
    """Shows a button to navigate to the next page based on session state."""
    next_button_text = f"Next → (Page {st.session_state.page + 1})" if "page" in st.session_state else "Next →"
    if st.button(next_button_text):
        next_page()

def show_previous_button():
    """Shows a button to navigate to the previous page."""
    if "page" in st.session_state and st.session_state.page > 1:
        prev_button_text = f"← Previous (Page {st.session_state.page - 1})"
        if st.button(prev_button_text):
            previous_page()


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
        padding: 0 10px;  /* Reduce padding around tooltip */
    }
    
    .emoji {
        font-size: 1.2em;  /* Slightly larger size for the emoji */
        margin-right: 5px;  /* Space between the emoji and the title */
        vertical-align: middle;  /* Align the emoji vertically with the text */
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
        margin-top: -5px;  /* Reduce the space between the tooltip and the text */
    }

    /* Make tooltip visible on hover over both emoji and title */
    .tooltip:hover .tooltiptext,
    .tooltip:hover {
        visibility: visible;
        opacity: 1;
    }

    /* Remove large gaps around the tooltip */
    p {
        margin: 0;  /* Remove default margin around the <p> tag */
        padding: 0;  /* Remove padding around the <p> tag */
    }
    </style>
    """, unsafe_allow_html=True)