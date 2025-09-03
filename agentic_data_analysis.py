
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Agentic Data Analyst", layout="centered")
st.title(" Agentic Data Analyst")
st.markdown("Upload your CSV and ask AI to analyze your dataset using LangChain agents.")

# --- API Key Input ---
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# --- CSV Upload ---
uploaded_file = st.file_uploader(" Upload your CSV File", type=["csv"])

# --- Tool Functions ---
def load_csv(file_path: str):
    df = pd.read_csv(file_path)
    return df.describe().to_string()

def detect_nulls(file_path: str):
    df = pd.read_csv(file_path)
    return df.isnull().sum().to_string()

def plot_data(file_path: str):
    df = pd.read_csv(file_path)
    sns.pairplot(df)
    plt.savefig("pairplot.png")
    return "Pairplot saved. Displaying below."

# --- Main Logic ---
if uploaded_file is not None:
    file_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    tools = [
        Tool(name="LoadCSV", func=load_csv, description="Load and describe a CSV file."),
        Tool(name="DetectNulls", func=detect_nulls, description="Detect missing values."),
        Tool(name="PlotData", func=plot_data, description="Generate pairplot from CSV.")
    ]

    llm = OpenAI(temperature=0,openai_api_key=openai_key)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    query = st.text_input(" Ask something about your data (e.g., 'Describe the dataset')")
    if st.button("Submit") and query:
        with st.spinner("Thinking..."):
            try:
                prompt = f"Using the file located at: {file_path}, {query}"
                response = agent.run(prompt)
                st.success(" Response received!")
                st.write(response)

                if "pairplot" in response.lower() or os.path.exists("pairplot.png"):
                    st.image("pairplot.png", caption="Pairplot Visualization")
            except Exception as e:
                st.error(f" Error: {e}")
else:
    st.info("Please upload a CSV file to continue.")