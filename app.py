import os
import streamlit as st
import PyPDF2
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title='DoctorDemma - AI Healthcare Assistant',
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
        font-family: 'Arial', sans-serif;
    }
    
    /* Custom card styling */
    .stCard {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers styling */
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    h2 {
        color: #34495e;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2rem;
        font-weight: 600;
    }
    
    h3 {
        color: #5d6d7e;
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.75rem;
        font-weight: 500;
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
        font-size: 1.1rem;
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
        font-size: 1.1rem;
        line-height: 1.5;
    }
    
    /* Input field styling */
    .stTextInput input {
        border-radius: 0.5rem;
        border: 2px solid #e9ecef;
        padding: 0.5rem;
        font-size: 1.1rem;
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
        font-size: 1.1rem;
    }
    
    /* Alert/Info box styling */
    .stAlert {
        border-radius: 0.5rem;
        border: none;
        padding: 1rem;
        font-size: 1.1rem;
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
        font-size: 1rem;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


# Load environment variables
load_dotenv()

# Constants
TEMPERATURE = 0.7
MAX_TOKENS = 1000
CONVERSATION_TIMEOUT = 30  # minutes

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = ""
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {
        "age": None,
        "gender": None,
        "medical_history": [],
        "current_medications": [],
        "allergies": [],
        "vital_signs": {},
        "consultation_datetime": None
    }
if "consultation_summary" not in st.session_state:
    st.session_state.consultation_summary = None

class MedicalAssistant:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.2-1b-preview",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Initialize chains with enhanced prompts
        self.diagnosis_chain = self._create_diagnosis_chain()
        self.follow_up_chain = self._create_follow_up_chain()
        self.summary_chain = self._create_summary_chain()
        
    def _create_diagnosis_chain(self) -> LLMChain:
        """Create an enhanced diagnostic chain with medical context."""
        prompt = PromptTemplate(
            input_variables=["symptoms", "history", "lab_report", "patient_data"],
            template="""
            Given the following patient information:
            - Age: {patient_data[age]}
            - Gender: {patient_data[gender]}
            - Medical History: {patient_data[medical_history]}
            - Current Medications: {patient_data[current_medications]}
            - Allergies: {patient_data[allergies]}
            - Vital Signs: {patient_data[vital_signs]}
            
            Current Symptoms: {symptoms}
            Previous Conversation: {history}
            Lab Reports: {lab_report}
            
            Please provide:
            1. Potential diagnoses (primary and differential)
            2. Severity assessment (Low/Medium/High)
            3. Recommended next steps
            4. Red flags to watch for
            5. Lifestyle recommendations
            
            Remember to maintain a professional tone and emphasize that this is an AI-generated preliminary assessment.
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)

    def _create_follow_up_chain(self) -> LLMChain:
        """Create an enhanced follow-up chain with context awareness."""
        prompt = PromptTemplate(
            input_variables=["follow_up", "history", "patient_data"],
            template="""
            Based on the patient profile:
            {patient_data}
            
            Previous conversation:
            {history}
            
            Follow-up question:
            {follow_up}
            
            Provide a clear, contextual response that:
            1. Directly addresses the question
            2. References relevant previous information
            3. Suggests additional clarifying questions if needed
            4. Maintains medical accuracy and appropriate disclaimers
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)

    def _create_summary_chain(self) -> LLMChain:
        """Create a chain for generating consultation summaries."""
        prompt = PromptTemplate(
            input_variables=["history", "patient_data"],
            template="""
            Generate a comprehensive consultation summary based on:
            
            Patient Information:
            {patient_data}
            
            Consultation History:
            {history}
            
            Please include:
            1. Chief complaints
            2. Key findings
            3. Preliminary diagnosis
            4. Recommendations
            5. Follow-up items
            
            Format the summary in a professional medical notation style.
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt)

def save_consultation_summary(summary: str, patient_data: Dict):
    """Save consultation summary and patient data to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_data = {
        "timestamp": timestamp,
        "patient_data": patient_data,
        "consultation_summary": summary
    }
    
    filename = f"consultation_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(summary_data, f, indent=4)
    
    return filename

def validate_vital_signs(vitals: Dict) -> tuple[bool, str]:
    """Validate vital signs are within reasonable ranges."""
    ranges = {
        "temperature": (35.0, 42.0),  # °C
        "heart_rate": (40, 200),      # bpm
        "blood_pressure_systolic": (70, 200),  # mmHg
        "blood_pressure_diastolic": (40, 130), # mmHg
        "oxygen_saturation": (70, 100),        # %
        "respiratory_rate": (8, 40)            # breaths per minute
    }
    
    for vital, (min_val, max_val) in ranges.items():
        if vital in vitals:
            if not min_val <= float(vitals[vital]) <= max_val:
                return False, f"{vital} is outside normal range"
    
    return True, "All vital signs within acceptable ranges"

def main():
    # Initialize medical assistant
    medical_assistant = MedicalAssistant(os.getenv("GROQ_API_KEY"))
    
    # Enhanced Sidebar with Patient Profile
    with st.sidebar:
        st.markdown("## 📋 Patient Profile")
        
        # Basic Information
        st.session_state.patient_data["age"] = st.number_input("Age", 0, 120, step=1)
        st.session_state.patient_data["gender"] = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        
        # Medical History
        with st.expander("🏥 Medical History"):
            history_input = st.text_area("Enter medical conditions (one per line)")
            st.session_state.patient_data["medical_history"] = [x.strip() for x in history_input.split("\n") if x.strip()]
        
        # Current Medications
        with st.expander("💊 Current Medications"):
            med_input = st.text_area("Enter medications (one per line)")
            st.session_state.patient_data["current_medications"] = [x.strip() for x in med_input.split("\n") if x.strip()]
        
        # Allergies
        with st.expander("⚠️ Allergies"):
            allergies_input = st.text_area("Enter allergies (one per line)")
            st.session_state.patient_data["allergies"] = [x.strip() for x in allergies_input.split("\n") if x.strip()]
        
        # Vital Signs
        with st.expander("📊 Vital Signs"):
            temp = st.number_input("Temperature (°C)", 35.0, 42.0, step=0.1)
            hr = st.number_input("Heart Rate (bpm)", 40, 200, step=1)
            bp_sys = st.number_input("Blood Pressure - Systolic", 70, 200, step=1)
            bp_dia = st.number_input("Blood Pressure - Diastolic", 40, 130, step=1)
            spo2 = st.number_input("Oxygen Saturation (%)", 70, 100, step=1)
            rr = st.number_input("Respiratory Rate (breaths/min)", 8, 40, step=1)
            
            st.session_state.patient_data["vital_signs"] = {
                "temperature": temp,
                "heart_rate": hr,
                "blood_pressure": f"{bp_sys}/{bp_dia}",
                "oxygen_saturation": spo2,
                "respiratory_rate": rr
            }
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 💬 Consultation")
        
        # Symptoms Input
        symptoms = st.text_area(
            "Describe Symptoms",
            height=150,
            placeholder="Please describe your symptoms in detail..."
        )
        
        # File Upload
        uploaded_file = st.file_uploader("Upload Medical Reports (PDF)", type="pdf")
        lab_report_content = ""
        if uploaded_file:
            lab_report_content = extract_pdf_content(uploaded_file)
            with st.expander("📄 Report Content"):
                st.text_area("Extracted Text", lab_report_content, height=200)
        
        # Consultation History
        if st.session_state.history:
            st.markdown("### Previous Messages")
            messages = st.session_state.history.split("\n\n")
            for msg in messages:
                st.markdown(f">{msg}")
    
    with col2:
        st.markdown("## 🎯 Actions")
        
        # Get Diagnosis
        if st.button("🔍 Generate Assessment", use_container_width=True):
            if symptoms or lab_report_content:
                with st.spinner("Analyzing information..."):
                    diagnosis = medical_assistant.diagnosis_chain.run(
                        symptoms=symptoms,
                        history=st.session_state.history,
                        lab_report=lab_report_content,
                        patient_data=st.session_state.patient_data
                    )
                    st.session_state.history = update_history(
                        st.session_state.history,
                        f"Symptoms: {symptoms}",
                        diagnosis
                    )
                    st.rerun()
        
        # Follow-up Questions
        st.markdown("### ❓ Follow-up Questions")
        follow_up = st.text_input("Enter your question")
        if st.button("Send Question", use_container_width=True):
            if follow_up:
                with st.spinner("Processing..."):
                    response = medical_assistant.follow_up_chain.run(
                        follow_up=follow_up,
                        history=st.session_state.history,
                        patient_data=st.session_state.patient_data
                    )
                    st.session_state.history = update_history(
                        st.session_state.history,
                        follow_up,
                        response
                    )
                    st.rerun()
        
        # Generate Summary
        if st.button("📝 Generate Consultation Summary", use_container_width=True):
            if st.session_state.history:
                with st.spinner("Generating summary..."):
                    summary = medical_assistant.summary_chain.run(
                        history=st.session_state.history,
                        patient_data=st.session_state.patient_data
                    )
                    st.session_state.consultation_summary = summary
                    
                    # Save summary to file
                    filename = save_consultation_summary(
                        summary,
                        st.session_state.patient_data
                    )
                    
                    st.download_button(
                        "⬇️ Download Summary",
                        summary,
                        file_name=filename,
                        mime="application/json"
                    )
                    
                    st.markdown(f"### Consultation Summary\n{summary}")

if __name__ == "__main__":
    main()