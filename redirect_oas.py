import streamlit as st
import streamlit.components.v1 as components

# TARGET URL
NEW_URL = "https://sudodocs-oas-validator.streamlit.app/"

st.set_page_config(page_title="Redirecting...", layout="wide")

# JavaScript to move them instantly
components.html(
    f"""
    <script>
        window.location.href = "{NEW_URL}";
    </script>
    """,
    height=0,
)

# Fallback text
st.title("We've Moved! 🚀")
st.markdown(f"The SudoDocs OAS Validator has a new home. [Click here if you are not redirected automatically.]({NEW_URL})")
