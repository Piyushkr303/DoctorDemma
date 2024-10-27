import os
import streamlit as st
import PyPDF2
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Must be the first Streamlit command
st.set_page_config(
    page_title='DoctorDemma - AI Healthcare Assistant',
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Custom card styling */
    .stCard {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #145c8e;
        transform: translateY(-2px);
    }
    
    /* Chat history styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        border-left: 5px solid #1f77b4;
    }
    
    /* Input field styling */
    .stTextInput input {
        border-radius: 0.5rem;
        border: 2px solid #e9ecef;
        padding: 0.5rem;
    }
    
    .stTextInput input:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    
    /* Alert/Info box styling */
    .stAlert {
        border-radius: 0.5rem;
        border: none;
        padding: 1rem;
    }
    
    /* Loading spinner styling */
    .stSpinner {
        text-align: center;
        color: #1f77b4;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        background-color: #f8f9fa;
        margin-top: 2rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

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
    """Format and append messages to chat history."""
    if not history:
        return f"User: {user_input}\nBot: {model_response}"
    return f"{history}\n\nUser: {user_input}\nBot: {model_response}"

def extract_pdf_content(pdf_file):
    """Extract and format text from PDF file."""
    try:
        content = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            content += page.extract_text() or ""
        return content.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def get_diagnosis(symptoms, history, lab_report):
    """Generate diagnosis with loading animation."""
    with st.spinner("🔄 Analyzing symptoms and generating diagnosis..."):
        return diagnosis_chain.run(symptoms=symptoms, history=history, lab_report=lab_report)

def handle_follow_up(follow_up_question, history):
    """Process follow-up questions with loading animation."""
    with st.spinner("🔄 Processing your question..."):
        return follow_up_chain.run(follow_up=follow_up_question, history=history)

# Header with custom styling
st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #1f77b4;'>🩺 DoctorDemma</h1>
        <p style='font-size: 1.2rem; color: #666;'>
            Your AI-powered healthcare assistant for preliminary diagnoses and medical insights
        </p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = ""

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h3 style='color: #1f77b4;'>📝 Patient Information</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Symptoms Input with enhanced styling
    symptoms = st.text_area(
        'Describe Your Symptoms',
        placeholder="E.g., I've been experiencing fever (38.5°C) for the past 3 days, accompanied by persistent dry cough and mild fatigue...",
        help="Provide detailed symptoms for more accurate diagnosis",
        height=150
    )
    
    st.markdown("---")
    
    # File Upload Section
    st.markdown("""
        <div style='text-align: center;'>
            <h4 style='color: #1f77b4;'>📄 Lab Reports</h4>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Medical Reports (PDF)",
        type="pdf",
        help="Upload relevant medical reports for better analysis"
    )
    
    lab_report_content = ""
    if uploaded_file:
        lab_report_content = extract_pdf_content(uploaded_file)
        with st.expander("📄 View Report Content"):
            st.text_area(
                "Extracted Content",
                value=lab_report_content,
                height=200,
                disabled=True
            )

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    # Chat History Section
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;'>
            <h3 style='color: #1f77b4;'>💬 Conversation History</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Display chat history with custom styling
    if st.session_state.history:
        messages = st.session_state.history.split("\n\n")
        for message in messages:
            st.markdown(f"""
                <div class="chat-message">
                    {message.replace('User:', '<strong style="color: #1f77b4;">User:</strong>').replace('Bot:', '<strong style="color: #2ecc71;">Bot:</strong>')}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👋 Start by describing your symptoms or uploading a lab report!")

with col2:
    # Action Buttons Section
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;'>
            <h3 style='color: #1f77b4;'>🎯 Actions</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Get Diagnosis", use_container_width=True):
        if symptoms or lab_report_content:
            diagnosis = get_diagnosis(symptoms, st.session_state.history, lab_report_content)
            st.session_state.history = update_history(st.session_state.history, symptoms, diagnosis)
            st.rerun()
        else:
            st.warning("⚠️ Please provide symptoms or upload a lab report.")
    
    st.markdown("---")
    
    # Follow-up Section
    st.markdown("<h4 style='color: #1f77b4;'>❓ Follow-up Questions</h4>", unsafe_allow_html=True)
    
    follow_up_question = st.text_input(
        "",
        placeholder="Ask any follow-up questions about your diagnosis..."
    )
    
    if st.button("📤 Send Question", use_container_width=True):
        if follow_up_question:
            follow_up_response = handle_follow_up(follow_up_question, st.session_state.history)
            st.session_state.history = update_history(st.session_state.history, follow_up_question, follow_up_response)
            st.rerun()
        else:
            st.warning("⚠️ Please enter your question.")

# Footer
st.markdown("""
    <div class='footer'>
        <p style='color: #666; font-size: 0.9rem;'>
            🔒 Your privacy is important to us. All conversations are confidential.<br>
            ⚠️ This AI assistant is for informational purposes only and should not replace professional medical advice.<br>
            © 2024 DoctorDemma - Powered by LangChain & Groq
        </p>
    </div>
""", unsafe_allow_html=True)
