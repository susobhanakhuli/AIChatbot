import os
import requests
import PyPDF2
import streamlit as st
from langchain.llms import OpenAI
from openai import OpenAI
from dotenv import load_dotenv

# Setting OpenAI API key as an environment variable
# Load environment variables from .env file
load_dotenv()

# Setting OpenAI API key from the environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Function to download PDF from a URL
def download_pdf_from_url(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        # st.success(f"PDF downloaded and saved to {output_path}")
    else:
        st.error(f"Failed to download PDF. Status code: {response.status_code}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to return answer of the question
def ask_question(text, question):
    """Asks a question about the provided text using OpenAI's API."""
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that can answer questions about the following text:\n\n{text}"},
            {"role": "user", "content": question},
        ],
        model="gpt-4o",
        temperature=1,
        max_tokens=7000,
        top_p=1
    )
    return response.choices[0].message.content

# Streamlit App Interface
st.title("PDF/URL Question-Answering with LangChain")

# Sidebar for file upload or URL input
st.sidebar.header("Input Options")
input_option = st.sidebar.radio("Choose input method:", ["Upload PDF", "Provide URL"])

document_text = ""

# Handle PDF upload
if input_option == "Upload PDF":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with open("temp_pdf.pdf", "wb") as f:
            f.write(uploaded_file.read())
        document_text = extract_text_from_pdf("temp_pdf.pdf")
        st.success("PDF uploaded and text extracted successfully!")

# Handle URL input
elif input_option == "Provide URL":
    url = st.sidebar.text_input("Enter the URL:")
    if url:
        try:
            download_pdf_from_url(url, "temp_pdf_from_url.pdf")
            document_text = extract_text_from_pdf("temp_pdf_from_url.pdf")
            st.success("URL content fetched and PDF extracted successfully!")
        except Exception as e:
            st.error(f"Error fetching PDF: {e}")

# Display extracted text and ask questions
if document_text:

    # Ask a question about the text
    question = st.text_input("Ask a question about the text:")
    if question:
        with st.spinner("Finding the answer..."):
            try:
                # agent = create_openai_agent()
                # answer = agent.run(question)
                answer = ask_question(document_text, question)
                st.write("### Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred while processing the question: {e}")