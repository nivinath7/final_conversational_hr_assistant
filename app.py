import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from utils import generate_follow_up_questions
import json

# --- LangChain Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Load Environment Variables ---
load_dotenv()

# --- Configurations ---
DOMAINS = {
    "Compensation & Performance": {
        "file": "compensation_performance.pdf",
        "json_file": "compensation_performance.json", # <-- ADD THIS LINE
        "icon": "💰",
        "description": "Salary, performance reviews, and career growth"
    },
    "Onboarding Assistant": {
        "file": "onboarding_assistant.pdf",
        "json_file": "onboarding_assistant.json", # <-- ADD THIS LINE
        "icon": "🚀",
        "description": "Welcome guide for new employees"
    },
    "Company Policies": {
        "file": "company_policies.pdf",
        "json_file": "company_policies.json", # <-- ADD THIS LINE
        "icon": "📋",
        "description": "Work policies, leave, and guidelines"
    },
    "Offboarding & Exit Process": {
        "file": "offboarding_exit.pdf",
        "json_file": "offboarding_exit.json", # <-- ADD THIS LINE
        "icon": "👋",
        "description": "Exit procedures and final settlements"
    },
    "Benefits & Eligibility": {
        "file": "benefits_eligibility.pdf",
        "json_file": "benefits_eligibility.json", # <-- ADD THIS LINE
        "icon": "🏥",
        "description": "Healthcare, insurance, and employee benefits"
    },
    "Payroll & Compliance": {
        "file": "payroll_compliance.pdf",
        "json_file": "payroll_compliance.json", # <-- ADD THIS LINE
        "icon": "📊",
        "description": "Payroll, taxes, and compliance matters"
    }
}

DOMAIN_QUESTIONS = {
    "Compensation & Performance": [
        "What is the annual performance review cycle timeline?",
        "How is my annual salary increment calculated?",
        "What are the different performance ratings and what do they mean?",
    ],
    "Onboarding Assistant": [
        "Where do I need to upload my official documents?",
        "How do I set up my company email and communication tools?",
        "When is the new hire orientation session?",
    ],
    "Company Policies": [
        "How many days of paid leave can I take per year?",
        "What is the process for travel expense reimbursement?",
        "What is the company's work-from-home policy?",
    ],
    "Offboarding & Exit Process": [
        "When will I receive my final settlement and payslip?",
        "What is the clearance process I need to follow?",
        "How do I apply for my experience and relieving letters?",
    ],
    "Benefits & Eligibility": [
        "What are the health insurance plans available to me?",
        "How do I enroll my family members in my health plan?",
        "Can I change my benefit selections mid-year?",
    ],
    "Payroll & Compliance": [
        "When is the deadline to submit investment proofs for tax savings?",
        "How is my Provident Fund (PF) contribution calculated?",
        "Where can I download my monthly payslips and annual Form 16?",
    ]
}

# --- Caching Functions ---
@st.cache_data
def load_knowledge_base(_file_path):
    reader = PdfReader(_file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

@st.cache_resource
def create_vector_store(_text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(_text_chunks, embedding=embeddings)
    return vector_store

@st.cache_data
def load_and_process_json(_file_path):
    """Loads a JSON file and converts its key-value pairs into searchable sentences."""
    if not os.path.exists(_file_path):
        return "" # Return empty string if JSON file doesn't exist
    
    with open(_file_path, 'r') as f:
        data = json.load(f)
    
    json_text = ""
    for key, value in data.items():
        formatted_key = key.replace('_', ' ')
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                json_text += f"Regarding {formatted_key}, the value for {sub_key.replace('_', ' ')} is {sub_value}. "
        else:
            json_text += f"The value for {formatted_key} is {value}. "
            
    return json_text

# --- Main App Logic ---
st.set_page_config(page_title="Optum HR Assistant", layout="wide")

# --- Enhanced CSS for better UI ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Landing page styling */
    .landing-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
        color: white;
    }
    
    .landing-title {
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: white; /* text color */
    background: #2563eb; /* blue background */
    display: inline-block; /* makes background wrap text instead of full width */
    padding: 0.5rem 1rem; /* some breathing space */
    border-radius: 0.5rem; /* optional: rounded corners */
    }

    
    .landing-subtitle {
        font-size: 1.3rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 3rem;
    }
    
    /* Domain cards styling */
    .domain-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 1.5rem;
        padding: 0 2rem 3rem 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .domain-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .domain-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        border-color: #003da1;
    }
    
    .domain-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .domain-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #003da1;
        margin-bottom: 0.5rem;
    }
    
    .domain-description {
        color: #666;
        font-size: 1rem;
        line-height: 1.4;
    }
    
    /* Chat interface styling */
    .chat-container {
        background: white;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Sidebar styling for chat mode */
    .css-1d391kg {
        background: linear-gradient(180deg, #003da1 0%, #0056d3 100%);
    }
    
    /* Button styling */
    div.stButton > button {
        background: linear-gradient(45deg, #003da1, #0056d3);
        border: none;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(45deg, #002d73, #003da1);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 61, 161, 0.3);
    }
    
    /* FAQ section styling */
    .faq-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #003da1;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid rgba(0, 61, 161, 0.1);
    }
    
    /* Header styling for chat mode */
    .chat-header {
        background: linear-gradient(90deg, #003da1, #0056d3);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .back-button {
        position: absolute;
        top: 1rem;
        left: 1rem;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

def reset_to_landing():
    """Reset the app to landing page state"""
    if 'selected_domain' in st.session_state:
        del st.session_state.selected_domain
    if 'messages' in st.session_state:
        st.session_state.messages = []
    if 'follow_up_questions' in st.session_state:
        st.session_state.follow_up_questions = []
    if 'conversation_chain' in st.session_state:
        del st.session_state.conversation_chain

# def select_domain(domain_name):
#     """Handle domain selection"""
#     st.session_state.selected_domain = domain_name
#     st.session_state.messages = []
#     st.session_state.follow_up_questions = []
    
#     # Load the knowledge base for the selected domain
#     with st.spinner(f"Preparing '{domain_name}'..."):
#         file_name = DOMAINS[domain_name]["file"]
#         kb_path = os.path.join("knowledge_base", file_name)
#         raw_text = load_knowledge_base(kb_path)
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         text_chunks = text_splitter.split_text(raw_text)
#         vector_store = create_vector_store(text_chunks)
#         memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
#         st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
#             llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
#             retriever=vector_store.as_retriever(),
#             memory=memory,
#             return_source_documents=True
#         )
def select_domain(domain_name):
    """Handle domain selection, loading both PDF and JSON."""
    st.session_state.selected_domain = domain_name
    st.session_state.messages = []
    st.session_state.follow_up_questions = []
    
    with st.spinner(f"Preparing '{domain_name}'..."):
        # --- START OF MODIFICATIONS ---
        
        # 1. Load PDF data
        pdf_file_name = DOMAINS[domain_name]["file"]
        pdf_path = os.path.join("knowledge_base", pdf_file_name)
        pdf_text = load_knowledge_base(pdf_path)
        
        # 2. Load and process JSON data
        json_file_name = DOMAINS[domain_name]["json_file"]
        json_path = os.path.join("knowledge_base", json_file_name)
        json_text = load_and_process_json(json_path)
        
        # 3. Combine text from both sources
        combined_text = pdf_text + "\n\n" + json_text
        
        # 4. Chunk the COMBINED text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_text(combined_text)
        
        # --- END OF MODIFICATIONS ---
        
        # The rest of the function proceeds as before
        vector_store = create_vector_store(text_chunks)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
        st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            retriever=vector_store.as_retriever(),
            memory=memory,
            return_source_documents=True
        )

# --- Chat handling function ---
# Moved this definition up here so it's defined before it's called
def handle_user_query(prompt):
    """Handle user queries and generate responses"""
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = st.session_state.conversation_chain({'question': prompt})
            bot_message = response['answer']
            sources = response['source_documents']
            
            st.markdown(bot_message)
            
            if sources:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(sources):
                        st.info(f"**Source {i+1}:**\n\n{source.page_content}")
    
    st.session_state.messages.append({"role": "assistant", "content": bot_message, "sources": sources})
    st.session_state.follow_up_questions = generate_follow_up_questions(prompt, bot_message)


# --- Check if we're on landing page or chat interface ---
if 'selected_domain' not in st.session_state:
    # LANDING PAGE
    st.markdown("""
    <div class="landing-header">
        <div class="landing-title">🤖 Optum HR Assistant</div>
        <div class="landing-subtitle">Your intelligent partner for all HR-related questions at Optum India</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create domain cards
    st.markdown('<div class="domain-grid">', unsafe_allow_html=True)
    
    cols = st.columns(3)
    for idx, (domain_name, domain_info) in enumerate(DOMAINS.items()):
        col_idx = idx % 3
        with cols[col_idx]:
            if st.button(
                f"{domain_info['icon']}\n\n{domain_name}\n\n{domain_info['description']}", 
                key=f"domain_{domain_name}",
                use_container_width=True,
                help=f"Click to access {domain_name} assistance"
            ):
                select_domain(domain_name)
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add some footer information
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="
            text-align: center; 
            background: #2563eb;   /* blue background */
            color: white;          /* white text */
            opacity: 0.95; 
            padding: 2rem; 
            border-radius: 0.5rem; 
            margin-top: 1rem;
        ">
            <p>💡 <strong>How it works:</strong> Select any HR domain above to start chatting with our AI assistant</p>
            <p>🔍 Get instant answers to your HR questions with relevant document sources</p>
            <p>🎯 Smart follow-up questions to help you get comprehensive information</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # CHAT INTERFACE
    # Sidebar with domain selector and back button
    with st.sidebar:
        if st.button("← Back to Home", key="back_home", use_container_width=True):
            reset_to_landing()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Select HR Domain")
        
        current_domain = st.session_state.selected_domain
        
        for domain_name, domain_info in DOMAINS.items():
            is_current = domain_name == current_domain
            button_style = "🔹" if is_current else ""
            
            if st.button(
                f"{button_style} {domain_info['icon']} {domain_name}", 
                key=f"sidebar_{domain_name}",
                use_container_width=True,
                disabled=is_current
            ):
                select_domain(domain_name)
                st.rerun()
    
    # Main chat interface
    domain_info = DOMAINS[st.session_state.selected_domain]
    
    # Header
    st.markdown(f"""
    <div class="chat-header">
        <h2>{domain_info['icon']} {st.session_state.selected_domain}</h2>
        <p style="margin: 0; opacity: 0.9;">{domain_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat and FAQ columns
    chat_col, faq_col = st.columns([2.5, 1])

    with chat_col:
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "follow_up_questions" not in st.session_state:
            st.session_state.follow_up_questions = []

        # Display chat messages
        for message in st.session_state.messages:
            avatar = "👤" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.info(f"**Source {i+1}:**\n\n{source.page_content}")
        
        # Follow-up questions
        if st.session_state.follow_up_questions:
            st.markdown("---")
            st.markdown("**💡 You might also want to ask:**")
            for question in st.session_state.follow_up_questions:
                if st.button(question, key=f"follow_up_{question}", use_container_width=True):
                    handle_user_query(question)
                    st.rerun()

        # Chat input
        if prompt := st.chat_input("Ask me anything about this HR domain..."):
            handle_user_query(prompt)
            st.rerun()

    with faq_col:
        # FAQ section with enhanced styling
        st.markdown('<div class="faq-container">', unsafe_allow_html=True)
        st.markdown("### 💡 Suggested Questions")
        st.markdown("---")
        
        questions = DOMAIN_QUESTIONS.get(st.session_state.selected_domain, [])
        for question in questions:
            if st.button(question, key=f"faq_{question}", use_container_width=True):
                handle_user_query(question)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)