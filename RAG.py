import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

#creating nessecerry objects of
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(embedding_function=embeddings)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, chunk_overlap=200, add_start_index=True
)
model = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    #api_key='togetherai_key',
    model='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
)
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Based on the following context, answer the question as accurately as possible "
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
)

#functions defining for pathing and retriving context
def extract_text_from_pdf(file_path):
    """Extract text from the uploaded PDF file using PyPDFLoader."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


# Streamlit UI
st.title("Question and answering from pdf")
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
question = st.text_input("Enter your question")

if st.button("Submit"):
    if pdf_file is not None and question:

      # Save the uploaded PDF file temporarily
        with open("temp_uploaded_file.pdf", "wb") as f:
            f.write(pdf_file.read())

        pdf_text = extract_text_from_pdf("/content/temp_uploaded_file.pdf")
        all_splits = text_splitter.split_documents(pdf_text)
        ids = vector_store.add_documents(documents=all_splits)

        results=retriever.batch([ question ])
        context = results[0][0].page_content  
        prompt = prompt_template.format(context=context, question=question)
        response = model.generate([prompt])

        response = f"You asked: '{question}'.\nHere's the extracted text from the PDF:\n\n{response.generations[0][0].text}..."

        st.text_area("Answer", response, height=300)
        # st.text_area("Chit Chat",topic_quote, height=300)
    else:
        st.warning("Please upload a PDF and enter a question.")
