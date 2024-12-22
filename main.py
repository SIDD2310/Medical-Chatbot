__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.path.append('/usr/bin/ffmpeg')
import streamlit as st
import os
import hashlib
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from textFunctions import get_pdf_text, get_pdfs_text, get_text_chunks

def hash_documents(pdf_docs):
    """Generate a hash for the content of uploaded documents."""
    hasher = hashlib.sha256()
    for doc in pdf_docs:
        hasher.update(doc.read())
        doc.seek(0)  # Reset file pointer after reading
    return hasher.hexdigest()

def init_ses_states():
    session_states = {
        "conversation": None,
        "chat_history": None,
        "pdf_analytics_enabled": False,
        "display_char_count": False,
        "display_word_count": False,
        "display_vaders": False,
        "api_authenticated": False
    }
    for state, default_value in session_states.items():
        if state not in st.session_state:
            st.session_state[state] = default_value

def get_vectorstore(text_chunks, doc_hash):
    """Load or create vectorstore based on the document hash."""
    persist_directory = f"./chroma_db/{doc_hash}"
    
    # Check if embeddings for the hashed documents already exist
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        print(f"Loading existing embeddings for hash: {doc_hash}")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))
    else:
        print(f"Creating new embeddings for hash: {doc_hash}")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
    
    return vectorstore

def get_conversation_chain(vectorstore, temp, model):
    llm = ChatOpenAI(temperature=temp, model_name=model)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def generate_summary(prompt_chain):
    combined_output = ""
    for prompt in prompt_chain:
        response = st.session_state.conversation({'question': prompt})
        output = response['chat_history'][-1].content
        combined_output += f"\n{output}\n"
    
    st.session_state.chat_history.append(combined_output)
    return combined_output

    # Generate summary
predefined_prompts_conflict = [
    """
    Medication History and Compliance:
    - Identify any discrepancies between:
      - Patient self-reports of medication adherence and documented records (e.g., progress notes, discharge summaries, pharmacy records).
      - Clinician observations and family/caregiver input.
      - Evidence of dose mismatches, skipped doses, or denial of medication during manic or depressive phases.
      Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,
    """
    Forensic History:
    - Highlight inconsistencies in legal or criminal history, such as:
      - Discrepancies between documents claiming no forensic history and records mentioning charges or convictions (e.g., shoplifting, trespassing).
      - Missing or incomplete details about reported legal incidents.
      Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,
    """
    Insight into Condition:
    - Identify conflicts in the patient's understanding of their mental health, including:
      - Patient denying symptoms or the need for treatment vs. clinical or family observations.
      - Contradictions between self-reported stability and documented functional impairments (e.g., job loss, hospitalization).
    Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,
    """
    Family Dynamics:
    - Explore discrepancies in family input, including:
      - Differing perspectives from caregivers (e.g., supportive vs. dismissive attitudes).
      - Conflicting accounts of family relationships, support levels, or the patient’s behavior during manic or depressive phases.
    Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,
    """
    Employment History:
    - Identify contradictions in the patient’s work history, such as:
      - Patient-reported job successes vs. employer reports of erratic behavior or dismissal.
      - Conflicting reasons for unemployment (e.g., voluntary resignation vs. termination).
      Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,
    """
    Psychiatric Diagnoses and Co-Morbidities:
    - Highlight inconsistencies in documented diagnoses or co-morbid conditions, including:
      - Variability in reported diagnoses across different records.
      - Conflicts between clinical notes, family input, and the patient’s self-reported symptoms.
      Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """
]

predefined_prompts_summary = [
    """### Identifying Information
    - Full Name: 
    - Date of Birth (DOB): 
    - Age (calculated age, noting deceased status if applicable): 
    - Gender: 
    - Marital Status: 
    - Occupation: 
    - Living Situation: 
    - Document Titles and Dates (list all uploaded documents with their titles and dates): 
    - Author (your name): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    
    """,

    """### Reason for Referral
    - Primary Concerns (describe the primary concerns or issues leading to the assessment or treatment): 
    - Referral Source (identify the referral source, e.g., self, family, GP, court): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Presenting Complaints
    - Current Symptoms (provide a detailed account of the patient’s most current symptoms, concerns, or challenges): 
    - Context/Triggers (include the context or triggers for recent episodes, if applicable): 
    - Duration and Progression (describe the duration and progression of symptoms): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Psychiatric History
    - Previous Diagnoses (organized by age at first diagnosis): 
      - Diagnosis: 
      - Age Diagnosed: 
    - Psychiatric Hospitalizations or Presentations:
      - Date or Age: 
      - Presenting Symptoms: 
      - Treatment Performed: 
      - Social Work Input: 
      - Medication Changes: 
      - Length of Admission: 
      - Significant Psychosocial Adjustments: 
      - Significant Medical Events: 
      Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Medical History
    - Past and Current Medical Conditions (include lab or diagnostic findings): 
    - Surgical History (list previous surgical procedures): 
    - Allergies and Sensitivities (include drug, food, or other allergies): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Medications
    - Current Medications (include name and dosage if known): 
    - Past Psychiatric Medications (include name, dosage, dates of dose changes, and side effects): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Substance Use History
    - Alcohol Use: 
    - Drug Use: 
    - Other Substances: 
    - Tobacco and Caffeine Consumption: 
    - Substance Use Treatment (include history of treatment or recovery efforts): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Family History
    - Psychiatric Conditions in Family: 
    - Medical Conditions in Family: 
    - Family Dynamics: 
    - Significant Life Events (e.g., history of abuse, neglect, trauma): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Developmental History
    - Early Childhood: 
    - Milestones (physical, social, emotional, cognitive): 
    - Academic History: 
    - Social History (childhood and adolescence):
    Only give the answer to the questions asked. Do not provide any additional information. 
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Social History
    - Educational History: 
    - Occupational History: 
    - Interpersonal Relationships: 
    - Hobbies and Interests: 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Forensic History
    - Legal Issues (record any forensic or legal matters): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Mental Status Examination (MSE)
    - Date of MSE: 
    - Appearance: 
    - Behavior: 
    - Speech: 
    - Mood and Affect:
      - Mood (patient's reported mood): 
      - Affect (observed affect): 
    - Thought Processes: 
    - Thought Content: 
    - Perceptions: 
    - Cognition: 
    - Insight and Judgment: 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Functional Assessment
    - Date of Functional Assessment: 
    - Daily Living Skills: 
    - Employment Capacity: 
    - Financial Management: 
    - Social Functioning: 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Psychological Testing
    - Testing Methods Used (e.g., IQ tests, personality assessments): 
    - Key Findings (results and interpretations): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Summary
    - Overall Summary and Impression: 
    Only give the answer to the questions asked. Do not provide any additional information.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """### Current Recommendations
    - Recommendations by Professionals:
      - Date: 
      - Professional’s Name/Title: 
      - Recommendations: 
    - Treatment Plan:
      - Medications (current medications and dosages): 
      - Psychological Interventions (therapies, counseling): 
      - Support Services (e.g., carers, mental health professionals, NDIS): 
      - Additional Services (if applicable): 
      Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """
]


def process_docs_and_generate_summary(pdf_docs, TEMP, MODEL):
    global predefined_prompts_summary
    """Processing documents and creating conversation chain."""
    st.session_state["conversation"] = None
    st.session_state["chat_history"] = []
    st.session_state["user_question"] = ""
    
    # Generate a unique hash for the uploaded documents
    doc_hash = hash_documents(pdf_docs)
    
    # Extract text and chunks
    raw_text = get_pdfs_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    
    # Generate or load embeddings based on the hash
    vectorstore = get_vectorstore(text_chunks, doc_hash)
    
    st.session_state.conversation = get_conversation_chain(vectorstore, temp=TEMP, model=MODEL)
    st.session_state.pdf_processed = True

    # Generate summary
    
    return generate_summary(predefined_prompts_summary)


def process_docs_and_generate_conflict(pdf_docs, TEMP, MODEL):
    global predefined_prompts_conflict
    """Processing documents and creating conversation chain."""
    st.session_state["conversation"] = None
    st.session_state["chat_history"] = []
    st.session_state["user_question"] = ""
    
    # Generate a unique hash for the uploaded documents
    doc_hash = hash_documents(pdf_docs)
    
    # Extract text and chunks
    raw_text = get_pdfs_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    
    # Generate or load embeddings based on the hash
    vectorstore = get_vectorstore(text_chunks, doc_hash)
    
    st.session_state.conversation = get_conversation_chain(vectorstore, temp=TEMP, model=MODEL)
    st.session_state.pdf_processed = True


    return generate_summary(predefined_prompts_conflict)



class OpenAIAuthenticator:
    @staticmethod
    def authenticate(api_key):
        if not api_key: return False
        os.environ['OPENAI_API_KEY'] = api_key
        try:
            llm = OpenAI()
            if llm("hi"): return True
            else: return False
        except Exception:
            return False

def api_authentication():
    openai_key = os.getenv("OPENAI_API_KEY")
    authenticator = OpenAIAuthenticator()
    if authenticator.authenticate(openai_key):
        st.session_state.api_authenticated = True
    elif not authenticator.authenticate(openai_key):
        st.session_state.api_authenticated = False

def chatbot_settings():
    global MODEL, TEMP
    with st.expander("Chat Bot Settings", expanded=True):
        MODEL = st.selectbox(label='Model', options=['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-0125'])
        TEMP = st.slider("Temperature", 0.0, 1.0, 0.7, help="Temperature for the model to generate responses. Lower values are more deterministic, higher values are more creative.")

def sidebar():
    global pdf_docs
    with st.sidebar:
        with st.expander("OpenAI API Authentication", expanded=True):
            api_authentication()
        chatbot_settings()
        with st.expander("Your Documents", expanded=True):
            pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)

def handle_userinput(user_question, prompt):
    # Run the chain, now it will only return 'answer'
    chain_output = st.session_state.conversation({'question': (prompt + user_question)})
    answer = chain_output['answer']
    
    # Retrieve documents separately
    source_docs = st.session_state.conversation.retriever.get_relevant_documents(user_question)

    st.session_state.chat_history = chain_output['chat_history']
    with st.spinner('Generating response...'):
        st.markdown(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

def process_docs(pdf_docs, TEMP, MODEL):
    st.session_state["conversation"] = None
    st.session_state["chat_history"] = None
    st.session_state["user_question"] = ""
    doc_hash = hash_documents(pdf_docs)
    raw_text = get_pdfs_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks, doc_hash)

    st.session_state.conversation = get_conversation_chain(vectorstore, temp=TEMP, model=MODEL)
    st.session_state.pdf_processed = True

def main():
    st.set_page_config(page_title="Multi-Document Chat Bot", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    global MODEL, TEMP, pdf_docs, predefined_prompts_summary, predefined_prompts_conflict
    init_ses_states()

    # Initialize session state for summary and conflict outputs
    if "summary_output" not in st.session_state:
        st.session_state["summary_output"] = ""
    if "conflict_output" not in st.session_state:
        st.session_state["conflict_output"] = ""

    deploy_tab, code_tab, test = st.tabs(["Note Synthesis", "Ambient AI Note Generation", "Testing"])
    
    with deploy_tab:
        st.title("Note Synthesis")
        c1, c2 = st.columns(2)
        sidebar()
        
        with c1:
            st.text_area(label="Prompt template for generating summary", value=predefined_prompts_summary, height=500)
            if pdf_docs and st.button("Generate Summary"):
                with st.spinner("Processing"):
                    summary = process_docs_and_generate_summary(pdf_docs, TEMP, MODEL)
                    st.session_state["summary_output"] = summary  # Save output in session state
            # Display summary output
            if st.session_state["summary_output"]:
                st.container().markdown(bot_template.replace("{{MSG}}", st.session_state["summary_output"]), unsafe_allow_html=True)
        
        with c2:
            st.text_area(label="Prompt template for generating conflicts", value=predefined_prompts_conflict, height=500)
            if pdf_docs and st.button("Generate Conflict"):
                with st.spinner("Processing"):
                    conflict = process_docs_and_generate_conflict(pdf_docs, TEMP, MODEL)
                    st.session_state["conflict_output"] = conflict  # Save output in session state
            # Display conflict output
            if st.session_state["conflict_output"]:
                st.container().markdown(bot_template.replace("{{MSG}}", st.session_state["conflict_output"]), unsafe_allow_html=True)
    
    with code_tab:
        import whisper
        from pydub import AudioSegment
        import tempfile
        import os
        import time

        # Set up the Streamlit interface
        st.title("Ambient AI Note Generation")

        # Allow the user to upload an MP3 file
        uploaded_file1 = st.file_uploader("Upload an MP3 file", type=["mp3"])

        if uploaded_file1 is not None:
            try:
                # Display a message to the user
                st.info("Processing your audio file, please wait...")

                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    temp_file.write(uploaded_file1.read())
                    temp_file_path = temp_file.name

                # Get the initial file size
                initial_file_size = os.path.getsize(temp_file_path) / (1024 * 1024)

                # Load the audio file using pydub
                audio = AudioSegment.from_file(temp_file_path, format="mp3")

                # Compress the audio file to the maximum extent possible
                compressed_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                audio.export(compressed_file_path, format="mp3", bitrate="32k")

                # Get the final file size
                final_file_size = os.path.getsize(compressed_file_path) / (1024 * 1024)

                # Display file sizes
                st.write(f"Initial file size: {initial_file_size:.2f} MB")
                st.write(f"Compressed file size: {final_file_size:.2f} MB")

                # Display a message indicating transcription is starting
                st.info("Transcribing your audio file, please wait...")

                # Perform transcription using the whisper model
                model = whisper.load_model("tiny")

                start_time = time.time()
                result = model.transcribe(compressed_file_path)
                transcription_time = time.time() - start_time

                # Display the transcription
                st.subheader("Transcription")
                st.write(result["text"])

                # Display transcription time
                st.write(f"Time taken to transcribe: {transcription_time:.2f} seconds")

                # Generate a summary using GPT model
                from openai import OpenAI

                client = OpenAI()

                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Summarize the following text: \n\n{result['text']}",
                        }
                    ],
                    model="gpt-4o-mini"
                )

                summary = chat_completion.choices[0].message.content

                st.subheader("Summary")
                st.write(summary)

            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
        else:
            st.write("Please upload an MP3 file to begin.")

            
    with test:
        st.title("Testing")
        if not pdf_docs:
            st.caption("Please Upload At Least 1 PDF")
            st.session_state.pdf_processed = False
        if st.sidebar.button("Process Files + New Chat"):
            if pdf_docs:
                with st.spinner("Processing"):
                    process_docs(pdf_docs, TEMP, MODEL)
            else: 
                st.caption("Please Upload At Least 1 PDF")
                st.session_state.pdf_processed = False
        PROMPT_TEMPLATE = "You are a helpful assistant"
        if st.session_state.get("pdf_processed") and st.session_state.api_authenticated:
            prompt = PROMPT_TEMPLATE
            with st.form("user_input_form"):
                user_question = st.text_input("Ask a question about your documents:")
                send_button = st.form_submit_button("Send")
            if send_button and user_question:
                handle_userinput(user_question, prompt)

if __name__ == '__main__':
    main()