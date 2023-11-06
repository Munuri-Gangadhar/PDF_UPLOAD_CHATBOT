import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import pickle
import os
from dotenv import load_dotenv

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = 'your_openai_api_key_here'

def main():
    st.header("Chat with your PDF file")

    pdf = st.file_uploader("Upload your PDF", type='pdf')
    st.write(f"PDF Uploaded: {pdf}")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.write(f"Extracted Text: {text}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            st.write("Vectorstore loaded from disk")
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
            st.write("Vectorstore computed and saved")

        query = st.text_input("Ask questions about the uploaded PDF file")
        st.write(f"Query: {query}")

        if query:
            docs = vectorstore.similarity_search(query=query, k=3)
            st.write(f"Docs: {docs}")

            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                st.write("Answer:", response)

if __name__ == "__main__":
    main()
