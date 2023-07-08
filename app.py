import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.agents import create_pandas_dataframe_agent
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.llms import OpenAI

# Define function to handle PDF file upload and text extraction
def process_pdf(file):
    reader = PdfReader(file)
    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # We need to split the text that we read into smaller chunks so that during information retrieval we don't hit the token size limits.
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    
    return docsearch, chain

# Define Streamlit app function
def app():
    st.set_page_config(page_title='Zee-Q 360', page_icon=':brain:')
    st.title('🧠 Zee-Q 360 AI Engine')
    st.write('Zee-Q 360 AI Engine is a question answering system that can answer questions based on the content of a PDF or CSV file.')
    
    key = st.text_input('Enter your OpenAI API key:')
    # OpenAI API Key
    os.environ['OPENAI_API_KEY'] = key
    
    file_option = st.radio("Select file type", ("PDF", "CSV"))
    file = st.file_uploader(f"Upload {file_option} file")
    
    if file_option == "PDF" and file is not None:
        docsearch, chain = process_pdf(file)

        i = 0
        while True:
            i += 1
            query = st.text_input(f'Enter your question {i}:', key=f'question_{i}')
            if not query:
                break

            docs = docsearch.similarity_search(query)
            response = chain.run(input_documents=docs, question=query)
            st.write("Answer:", response)

    elif file_option == "CSV" and file is not None:
        df = pd.read_csv(file)
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

        i = 0
        while True:
            i += 1
            query = st.text_input(f'Enter your question {i}:', key=f'question_{i}')
            if not query:
                break

            response = agent.run(query)
            st.write("Answer:", response)

if __name__ == '__main__':
    app()
