import os
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from apikey import apikey

os.environ["OPENAI_API_KEY"] = apikey

def app():
    st.title("CSV Query app")
    st.write("upload a csv and enter query to get an answer")
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if not file:
        st.stop()

    data = pd.read_csv(file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data, verbose=True)

    query = st.text_input("Enter a query")

    if st.button("Execute"):
        answer = agent.run(query)
        st.write("Answer:")
        st.write(answer)


if __name__== "__main__":
    app()


