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
    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title. Maintain provided headings and subheadings.:
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
    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title. Maintain provided headings and subheadings.:
    Forensic History:
    - Highlight inconsistencies in legal or criminal history, such as:
      - Discrepancies between documents claiming no forensic history and records mentioning charges or convictions (e.g., shoplifting, trespassing).
      - Missing or incomplete details about reported legal incidents.
      Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,
    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title. Maintain provided headings and subheadings.:
    Insight into Condition:
    - Identify conflicts in the patient's understanding of their mental health, including:
      - Patient denying symptoms or the need for treatment vs. clinical or family observations.
      - Contradictions between self-reported stability and documented functional impairments (e.g., job loss, hospitalization).
    Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,
    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title. Maintain provided headings and subheadings.:
    Family Dynamics:
    - Explore discrepancies in family input, including:
      - Differing perspectives from caregivers (e.g., supportive vs. dismissive attitudes).
      - Conflicting accounts of family relationships, support levels, or the patient’s behavior during manic or depressive phases.
    Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,
    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title. Maintain provided headings and subheadings.:
    Employment History:
    - Identify contradictions in the patient’s work history, such as:
      - Patient-reported job successes vs. employer reports of erratic behavior or dismissal.
      - Conflicting reasons for unemployment (e.g., voluntary resignation vs. termination).
      Only give the answer to the questions asked. Do not provide any additional information.
      Write it in short and in bullet points.
      If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
      Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,
    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title. Maintain provided headings and subheadings.:
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
    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Identifying Information
    - Full Name: 
    - Date of Birth (DOB): 
    - Age (Only include number): 
    - Gender: 
    - Marital Status: 
    - Occupation: 
    - Living Situation: 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:

    ### Reason for Referral
    - Primary Concerns (describe the primary concerns or issues leading to the assessment or treatment): 
    - Referral Source (identify the referral source, e.g., self, family, GP, court, and include the date of referral): 

    Only give the answer to the questions asked. Do not provide any additional information.  
    Write it in short and in bullet points.  
    If you don’t know the answer or if the provided context does not provide specific details or information, then don’t give any answer. Skip it.  
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Presenting Complaints
    - Current Symptoms (provide a detailed account of the patient’s most current symptoms, concerns, or challenges): 
    - Context/Triggers (include the context or triggers for recent episodes, if applicable): 
    - Duration and Progression (describe the duration and progression of symptoms): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Psychiatric History
    - Previous Diagnoses (organized by age at first diagnosis): 
      - Diagnosis: 
      - Age Diagnosed: 
    - Psychiatric Hospitalizations or Presentations:  
        Provide the information in paragraph form. Address each of the following points if the details are available:  
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

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Medical History
    - Past and Current Psychiatric and Medical conditions (include lab or diagnostic findings and also include dates with each): 
    - Surgical History (list previous surgical procedures): 
    - Allergies and Sensitivities (include drug, food, or other allergies): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    ### Medications
    - Current Medications (include name and dosage if known): 
    - Past Psychiatric Medications (include name, dosage, dates of dose changes, and side effects): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    Follow the template below
    ### Substance Use History
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

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Family History
    - Psychiatric Conditions in Family: 
    - Medical Conditions in Family: 
    - Family Dynamics: 
    - Significant Life Events (e.g., history of abuse, neglect, trauma): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Developmental History
    - Early Childhood: 
    - Milestones (physical, social, emotional, cognitive): 
    - Academic History: 
    - Social History (childhood and adolescence):
    Only give the answer to the questions asked. Do not provide any additional information. 
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    ### Social History
    - Educational History: 
    - Occupational History: 
    - Interpersonal Relationships: 
    - Hobbies and Interests: 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Forensic History
    - Legal Issues (record any forensic or legal matters): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Mental Status Examination (MSE)
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

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Functional Assessment
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

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Psychological Testing
    - Testing Methods Used (e.g., IQ tests, personality assessments): 
    - Key Findings (results and interpretations): 
    Only give the answer to the questions asked. Do not provide any additional information.
    Write it in short and in bullet points.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Summary
    - Overall Summary and Impression:
    Write the heading as Summary and then provide the answer to the questions asked. Do not provide any additional information. 
    Only give the answer to the questions asked. Do not provide any additional information.
    If you dont know the answer or if the provided context does not provide specific details or information then dont give any answer. Skip it.
    Give the headings and subheadings as they are and only provide the answers to the questions asked.
    """,

    """You are a mental health professional tasked with writing notes. The report should include the following sections. If the information is not available, skip the section. Include the title:
    
    ### Current Recommendations
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


def handle_userinput_1(user_question, prompt):
    response = st.session_state.conversation({'question': (prompt+user_question)})
    st.session_state.chat_history = response['chat_history']
    with st.spinner('Generating response...'):
        display_convo(prompt)
        

def display_convo(prompt):
    with st.container():
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(user_template.replace("{{MSG}}", message.content[len(prompt):]), unsafe_allow_html=True)
                
                
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

    deploy_tab, code_tab, test = st.tabs(["Note Synthesis", "Ambient AI Note Generation", "Treatment"])
    
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
        import tempfile
        import whisper
        import datetime
        import subprocess
        import wave
        import contextlib
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
        from pyannote.audio import Audio
        from pyannote.core import Segment
        embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
        audio_processor = Audio()
        prompt_template = """Generate a professional therapy progress note based on the following session transcript. The
note must strictly adhere to the outlined structure below and include only information
explicitly discussed in the session. Summarize disparate ideas within each heading into
cohesive paragraphs wherever possible but use bullet points sparingly for clarity when
necessary. Do not add interpretations, assumptions, or any extra material beyond what is
stated in the session. Write in a concise, professional tone suitable for an electronic medical
record (EMR).
Structure:
1. Presenting Problem:
o A brief summary of the client&#39;s main concerns or reasons for the session.
Include any relevant quotes from the client that capture the presenting issue.

2. Client History:
o Summarize any background details provided by the client that give context to
the presenting problem.
o Address contributing factors, but minimize bullet points unless they clarify
distinct issues.
3. Key Issues Discussed:
o Detail the specific problems or challenges discussed during the session.
Include client quotes where appropriate to highlight key themes or
expressions.

4. Cognitive Themes Identified:
o Summarize thought patterns or recurring themes that emerged in the session.
o Use client quotes selectively to illustrate significant insights or challenges.
5. Therapeutic Focus:
o Describe the strategies, techniques, or frameworks introduced or explored
during the session.
o Combine ideas into paragraphs but note multiple approaches clearly.
6. Interventions:
o Detail the specific interventions or techniques applied in the session.
o Use bullet points only for distinct actions or steps taken.
7. Homework Assigned:
o Clearly outline any tasks, exercises, or self-reflection activities assigned to the
client.
8. Plan:
o Outline the therapeutic goals for the next session or ongoing treatment.

Example Output:
[Presenting Problem:
The client presented with persistent feelings of sadness, lack of motivation, and a sense of
being &quot;stuck&quot; in their daily life. They described dissatisfaction with previously fulfilling
tasks, stating, &quot;I just feel really stuck and not happy.&quot;
Client History:
The client is a stay-at-home parent of two children, aged 8 and 10, and has been in this role
for 10 years. They historically found meaning and fulfillment in household and childcare
responsibilities. However, over the past six months, they reported a significant decline in
motivation and purpose, coinciding with their youngest child starting school.

Additional contributing factors include:
 Increased time alone and reduced social interactions, noting that former neighborhood
friendships have &quot;fizzled out.&quot;
 A reduced sense of accomplishment and reliance on their spouse to manage tasks they
previously handled independently.
Key Issues Discussed:
The client struggles with routine, often finding it difficult to get out of bed due to feelings of
pointlessness and emotional avoidance. Staying in bed provides temporary relief but leads to
guilt and frustration. They stated, &quot;I feel frustrated with myself because I know what might
help, but I just don’t feel motivated to do it.&quot;
They acknowledged perceived criticism from family members as a significant stressor. For
example, their mother pointed out they are &quot;not doing as much,&quot; which they found hurtful but
true. Similarly, the client’s husband has expressed frustration with the additional
responsibilities he has assumed.
Cognitive Themes Identified:
 Maladaptive thoughts, such as &quot;What’s the point?&quot; and &quot;It won’t help anyway.&quot;
 Recognition of a negative cycle of avoidance, where efforts to delay emotional pain
lead to increased guilt and dissatisfaction. The client reflected, &quot;I know I’m delaying
the pain, but it’s just so hard to break out of it.&quot;
Therapeutic Focus:
Cognitive behavioral strategies were introduced to address unhelpful thought patterns. I
emphasized the principle that behavioral activation—taking action even without immediate
emotional reward—can help initiate positive change. Household tasks were reframed as
opportunities for potential satisfaction, even if immediate rewards are absent.
The client was encouraged to focus on small, manageable goals to build momentum. They
agreed, &quot;If I don’t do anything, I’m not going to feel any better.&quot; Mindfulness techniques were
introduced to help the client stay present during daily tasks and avoid rumination.
Interventions:
 Developed adaptive self-talk strategies, including &quot;I can do this&quot; and &quot;I’m strong
enough to push through.&quot;
 Suggested starting each day with one functional task, such as waking up on time and
preparing for the day.
 Highlighted the importance of recognizing progress without judgment and reinforcing
that &quot;The probability of feeling better is higher if you try than if you don’t.&quot;
Homework Assigned:
The client was tasked with practicing adaptive self-talk upon waking and attempting to wake
up on time and follow their morning routine during the next school week. They were
encouraged to approach progress with self-compassion, recognizing that setbacks do not
negate their overall progress.

Plan:
I will monitor the client’s progress with adaptive responses and behavioral activation in the
next session. We will explore deeper emotional underpinnings of depressive symptoms if
necessary and introduce additional strategies to rebuild a sense of purpose and fulfillment as
appropriate.
]
Strictly adhere to this structure for consistency and clarity.
                    """
        # Helper function to convert seconds to time
        def time(secs):
            return str(datetime.timedelta(seconds=round(secs)))

        def process_audio(path, num_speakers, model_size):
            # Convert to WAV if necessary and ensure mono audio
            if path[-3:] != 'wav':
                subprocess.call(['ffmpeg', '-i', path, '-ac', '1', 'audio.wav', '-y'])
                path = 'audio.wav'
            else:
                # Ensure mono conversion for WAV files
                subprocess.call(['ffmpeg', '-i', path, '-ac', '1', 'audio_mono.wav', '-y'])
                path = 'audio_mono.wav'

            # Load Whisper model
            model = whisper.load_model(model_size)

            # Transcribe audio
            result = model.transcribe(path)
            segments = result["segments"]

            # Get duration of audio
            with contextlib.closing(wave.open(path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)

            # Extract embeddings for each segment
            def segment_embedding(segment):
                start = segment["start"]
                end = min(duration, segment["end"])
                clip = Segment(start, end)
                waveform, _ = audio_processor.crop(path, clip)
                return embedding_model(waveform[None])

            embeddings = np.zeros((len(segments), 192))
            for i, segment in enumerate(segments):
                embeddings[i] = segment_embedding(segment)

            embeddings = np.nan_to_num(embeddings)

            # Cluster embeddings into speakers
            clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
            labels = clustering.labels_
            for i in range(len(segments)):
                segments[i]["speaker"] = f'SPEAKER {labels[i] + 1}'

            # Generate transcript with speaker labels
            transcript = ""
            for i, segment in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    transcript += f"\n{segment['speaker']} {time(segment['start'])}\n"
                transcript += segment["text"].strip() + " "

            return transcript

        st.title("Ambient AI Note Generation")
        
        # Add a sidebar for user options
        st.header("Input Options")
        input_option = st.radio("Choose an input option:", ("Upload Audio File", "Upload Transcription"))

        if input_option == "Upload Audio File":
            # Allow the user to upload an MP3 file
            uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
            num_speakers = st.number_input("Number of Speakers", min_value=1, max_value=10, value=2, step=1)
            model_size = st.selectbox("Model Size", options=["tiny", "base", "small", "medium", "large"])

            if uploaded_file is not None:
                with open("temp_audio", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Processing audio..."):
                    transcript = process_audio("temp_audio", num_speakers, model_size)

                st.success("Processing complete!")
                st.text_area("Transcript", transcript, height=400)
                    # Generate a summary using GPT model
                from openai import OpenAI

                client = OpenAI()

                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt_template} \n\n{transcript}",
                        }
                    ],
                    model="gpt-4o-mini"
                )

                summary = chat_completion.choices[0].message.content

                st.subheader("Summary")
                st.write(summary)

    

        elif input_option == "Upload Transcription":
            # Allow the user to upload or paste a transcription
            transcription_input = st.text_area("Paste your transcription here:", "")

            
            st.text_area(label="Modify the prompt here" ,value=prompt_template)
            
            if transcription_input:
                try:
                    # Display the transcription
                    st.subheader("Transcription")
                    st.markdown(transcription_input)
                    

                    # Generate a summary using GPT model
                    from openai import OpenAI

                    client = OpenAI()

                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": f"{prompt_template} \n\n{transcription_input}",
                            }
                        ],
                        model="gpt-4o-mini"
                    )

                    summary = chat_completion.choices[0].message.content

                    st.subheader("Summary")
                    st.markdown(summary)

                except Exception as e:
                    # Handle any errors that occur
                    st.error(f"An error occurred: {e}")
            else:
                st.write("Please paste a transcription to begin.")



            
    with test:
        st.title("Treatment")
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
            with st.form("user_input_for"):
                user_question = st.text_input("Ask a question about your documents:")
                send_button = st.form_submit_button("Send")
            if send_button and user_question:
                handle_userinput_1(user_question, prompt)


            
if __name__ == '__main__':
    main()