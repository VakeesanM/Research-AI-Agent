import streamlit as st
from research_assistant import get_brief
from dotenv import load_dotenv
import os
load_dotenv()
os.getenv("OPENAI_API_KEY")


st.set_page_config(page_title="AI Research Assistant", page_icon="📝", layout="wide")
st.header("AI Research Assistant Demo - Vakeesan M")
st.write("This is a Demo for automonous ReAct AI Research Assistant. The main purpose of this agent, is not do the research for you but rather to create a research brief and provide resources(Academic Papers and Articles) to use as sources for your own research.")
query = st.text_area("Enter the topic you would like to research. Please Be specific. ")

if st.button("Submit"):
    with st.spinner("Creating Research Brief...(May take up to 2 Minutes in the Worst Case)", show_time=True):
        st.write(get_brief(query))




st.subheader("Acknowledgement")
"Thank you to arXiv for use of its open access interoperability."
