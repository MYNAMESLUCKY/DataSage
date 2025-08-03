import streamlit as st

st.title("Test App")
st.write("This is a simple test")

if st.button("Test Button"):
    st.success("Button works!")