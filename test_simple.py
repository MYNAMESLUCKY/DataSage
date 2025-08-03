import streamlit as st

st.title("Test RAG System")
st.write("This is a simple test to check if Streamlit is working properly")

if st.button("Test Button"):
    st.success("Button works!")

st.write("Testing session state...")
if 'counter' not in st.session_state:
    st.session_state.counter = 0

if st.button("Increment"):
    st.session_state.counter += 1

st.write(f"Counter: {st.session_state.counter}")