import streamlit as st
import PyPDF2
import requests
from transformers import pipeline

# Load a question-answering pipeline
qa_pipeline = pipeline("question-answering")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to fetch text from a URL
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            st.error("Failed to fetch the URL content.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit App
st.title("PDF/URL Question-Answering Tool")
st.write("Upload a PDF or provide a link, then ask questions to get answers.")

# Sidebar for file upload or URL input
st.sidebar.header("Input Options")
input_option = st.sidebar.radio("Choose input method:", ["Upload PDF", "Provide URL"])

# Variable to store the extracted text
document_text = ""

# PDF Upload
if input_option == "Upload PDF":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        document_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF uploaded and text extracted successfully!")

# URL Input
elif input_option == "Provide URL":
    url = st.sidebar.text_input("Enter the URL:")
    if url:
        document_text = fetch_text_from_url(url)
        if document_text:
            st.success("URL content fetched successfully!")

# Text Display and Question-Answering
if document_text:
    st.text_area("Extracted Text:", document_text[:1000] + "..." if len(document_text) > 1000 else document_text, height=200)

    # Question Input
    question = st.text_input("Ask a question about the text:")
    if question:
        with st.spinner("Finding the answer..."):
            try:
                answer = qa_pipeline(question=question, context=document_text)
                st.write("### Answer:")
                st.write(answer["answer"])
            except Exception as e:
                st.error(f"An error occurred while processing the question: {e}")
