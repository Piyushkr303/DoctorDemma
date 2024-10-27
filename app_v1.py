import os
import streamlit as st
import PyPDF2  # Use PyPDF2 for PDF reading
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ Groq API Key is missing.")
    st.stop()

# Initialize the ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-1b-preview")

# Define Prompts
diagnosis_prompt = PromptTemplate(
    input_variables=["symptoms", "history", "lab_report"],
    template=(
        "Based on the following conversation history: {history}, "
        "the patient presents these symptoms: {symptoms}. "
        "Lab report summary: {lab_report}. Provide a diagnosis."
    ),
)

follow_up_prompt = PromptTemplate(
    input_variables=["follow_up", "history"],
    template=(
        "Given the following conversation history: {history}, "
        "respond to the user's follow-up question: {follow_up}. "
        "Keep the answer concise and relevant."
    ),
)

# Initialize LangChain Agents
diagnosis_chain = LLMChain(llm=llm, prompt=diagnosis_prompt)
follow_up_chain = LLMChain(llm=llm, prompt=follow_up_prompt)

def update_history(history, user_input, model_response):
    """Append user input and model response to the conversation history."""
    return f"{history}\nUser: {user_input}\nBot: {model_response}"

def extract_pdf_content(pdf_file):
    """Extract text from an uploaded PDF file using PyPDF2."""
    content = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        content += page.extract_text() or ""  # Safely append text
    return content

def get_diagnosis(symptoms, history, lab_report):
    """Generate diagnosis using LangChain."""
    with st.spinner("Generating diagnosis..."):
        return diagnosis_chain.run(symptoms=symptoms, history=history, lab_report=lab_report)

def handle_follow_up(follow_up_question, history):
    """Handle follow-up questions and generate responses."""
    with st.spinner("Processing your question..."):
        return follow_up_chain.run(follow_up=follow_up_question, history=history)

# Streamlit UI Setup
st.set_page_config(page_title='Healthcare Chatbot', page_icon="🩺", layout="wide")

# Header
st.title('🩺 DoctorDemma')
st.markdown("Use this chatbot to get diagnoses based on symptoms or lab reports, and follow up with further questions.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = ""

# Sidebar Input Section
with st.sidebar:
    st.header("📝 Patient Information")
    
    symptoms = st.text_input(
        'Enter Symptoms', 
        placeholder="E.g., Fever, Cough, Headache", 
        help="Describe symptoms clearly to get an accurate diagnosis."
    )
    
    uploaded_file = st.file_uploader("Upload Lab Report (PDF)", type="pdf", help="Upload a PDF containing lab results.")
    
    lab_report_content = ""
    if uploaded_file:
        lab_report_content = extract_pdf_content(uploaded_file)
        st.text_area("📄 Lab Report Content:", value=lab_report_content, height=150)

# Chat History Section
st.subheader("📜 Chat History")
st.text_area(
    "Conversation", 
    value=st.session_state.history, 
    height=300, 
    disabled=True, 
    help="See the conversation history here."
)

# Diagnosis Section
st.markdown("---")
st.header("Get a Diagnosis")

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("Get Diagnosis", use_container_width=True):
        if symptoms:  # Check if symptoms are provided
            diagnosis = get_diagnosis(symptoms, st.session_state.history, lab_report_content)
            st.session_state.history = update_history(st.session_state.history, symptoms, diagnosis)
            st.rerun()  # Use rerun() instead of experimental_rerun()
        else:
            st.warning("⚠️ Please provide symptoms for diagnosis.")

with col2:
    st.info("💡 Provide either symptoms or a lab report for diagnosis.")

# Follow-up Question Section
st.markdown("---")
st.header("Ask a Follow-up Question")

follow_up_question = st.text_input(
    "Enter your follow-up question...", 
    placeholder="E.g., What medication should I take?"
)

if st.button("Submit Follow-up", use_container_width=True):
    if follow_up_question:
        follow_up_response = handle_follow_up(follow_up_question, st.session_state.history)
        st.session_state.history = update_history(st.session_state.history, follow_up_question, follow_up_response)
        st.rerun()  # Use rerun() instead of experimental_rerun()
    else:
        st.warning("⚠️ Please enter a question before submitting.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>© 2024 Healthcare Chatbot - Powered by LangChain & Groq</div>", 
    unsafe_allow_html=True
)