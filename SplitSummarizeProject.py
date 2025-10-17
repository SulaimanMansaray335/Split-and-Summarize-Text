import os 
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import pandas as pd
import io 
import chardet  
from pypdf import PdfReader
from langchain_core.documents import Document

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]   

def load_LLM(openai_api_key):
    llm = OpenAI(temperature = 0, openai_api_key=openai_api_key)
    return llm



st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

col1, col2 = st.columns(2) 

with col1:
    st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app")
    
with col2:
    st.write("Contact Tariq Mansaray tariq.mansaray@gmail.com for your AI solutions")
    
st.markdown("## Enter Your OpenAI API Key")

def get_openai_api_key():
    input_text = st.text_input(label= "OpenAI API Key", placeholder = "Ex: sk-2twmA8tfCb8un4...", key = "openai_api_key_input", type = "password")
    return input_text

openai_api_key = get_openai_api_key()

st.markdown("## Upload the text file you want to summarize")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])


st.markdown("### Here is your summary")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    
    string_data = stringio.read()
    
    file_input = string_data
    
    if len(file_input.split(" ")) > 20000:
        st.write("Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)",
                 icon = "⚠️")
        st.stop() 
        
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n"],
        chunk_size = 5000,
        chunk_overlap = 350
    )
    
    splitted_documents = text_splitter.split_documents([file_input])
    
    llm = load_LLM(openai_api_key = openai_api_key)
    
    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type= "map_reduce"
    )
    
    summary_output = summarize_chain.run(splitted_documents) 
    
    st.write(summary_output)