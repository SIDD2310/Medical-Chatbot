import streamlit as st
from dotenv import load_dotenv
import os
import openai
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

from htmlTemplates import css, bot_template, user_template
from codeDisplay import display_code
from textFunctions import get_pdf_text, get_pdfs_text, get_text_chunks




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


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
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


def handle_userinput(user_question, prompt):
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


def process_docs(pdf_docs, TEMP, MODEL):
    st.session_state["conversation"] = None
    st.session_state["chat_history"] = None
    st.session_state["user_question"] = ""

    raw_text = get_pdfs_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)

    st.session_state.conversation = get_conversation_chain(vectorstore, temp=TEMP, model=MODEL)
    st.session_state.pdf_processed = True



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
    # load_dotenv()
    openai_key = st.secrets["OPENAI_API_KEY"]
    # openai_key = os.getenv("OPENAI_API_KEY")
    # st.write(openai_key)
    # if not st.session_state.api_authenticated:
    #     openai_key = st.text_input("OpenAI API Key:", type="password")
    #     if not openai_key:
    #         st.info("Please enter your API Key.")
    #         return
    authenticator = OpenAIAuthenticator()
    if authenticator.authenticate(openai_key):
        st.session_state.api_authenticated = True
    elif not authenticator.authenticate(openai_key):
        st.session_state.api_authenticated = False
    
    #     if st.session_state.api_authenticated:
    #         st.success("Authentication Successful!")
    #     else:
    #         st.error("Invalid API Key. Please try again.")
    # else:
    #     st.success("Authentication Successful!")


def chatbot_settings():
    global MODEL, PROMPT_TEMPLATE, TEMP
    with st.expander("Chat Bot Settings", expanded=True):
        MODEL = st.selectbox(label='Model', options=['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo-0125'])
        TEMP = st.slider("Temperature", 0.0, 1.0, 0.5, help="Temperature for the model to generate responses. Lower values are more deterministic, higher values are more creative.")
    


def sidebar():
    global pdf_docs
    with st.sidebar:
        with st.expander("OpenAI API Authentication", expanded=True):
            api_authentication()
        chatbot_settings()
        with st.expander("Your Documents", expanded=True):
            pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
            if st.button("Process Files + New Chat"):
                if pdf_docs:
                    with st.spinner("Processing"):
                        process_docs(pdf_docs, TEMP, MODEL)
                else: 
                    st.caption("Please Upload At Least 1 PDF")
                    st.session_state.pdf_processed = False


def main():
    st.set_page_config(page_title="Multi-Document Chat Bot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    global MODEL, PROMPT_TEMPLATE, TEMP, pdf_docs
    init_ses_states()
    deploy_tab, code_tab= st.tabs(["Deployment", "Code"])
    with deploy_tab:
        st.title(":books: Medical-Document ChatBot 🤖")
        sidebar()
        PROMPT_TEMPLATE = st.text_area(
            label="Prompt Template", 
            value='''You are a helpful, respectful, and honest assistant dedicated to providing informative and accurate response based on provided context only. You don't derive answer outside context, while answering your answer should be precise, accurate, clear and should not be verbose and only contain answer. In context you will have texts which is unrelated to question, please ignore that context only answer from the related context only. If the question is unclear, incoherent, or lacks factual basis, please clarify the issue rather than generating inaccurate information. If formatting, such as bullet points, numbered lists, tables, or code blocks, is necessary for a comprehensive response, please apply the appropriate formatting.

            Please provide the answer in the following format.
            
            Identifying Information
            - Full name
            - Date of birth
            - Age 
            - Gender
            - Marital status
            - Occupation
            - Living situation
            - Date of the report
            - Author

            Reason for Referral
            - Primary concerns leading to assessment or treatment.
            - Source of referral (e.g., self, family, GP, court).

            Presenting Complaints
            - Patient’s current symptoms, concerns, and challenges.
            - Context or triggers for the recent episode (if applicable).

            Psychiatric History
            - Previous diagnoses and treatments.
            - History of psychiatric hospitalizations.
            - Episodes of mania, depression, anxiety, psychosis, etc.
            - History of suicidal ideation or attempts.
            - History of medication use and adherence.

            Medical History
            - Past and current medical conditions.
            - Medications (psychiatric and non-psychiatric).
            - Allergies, substance use, and relevant lab results.

            Developmental History
            - Prenatal and birth details.
            - Early childhood milestones.
            - Academic, social, and emotional development.

            Family History
            - Psychiatric and medical conditions in family members.
            - Family dynamics and relationships.
            - Any history of abuse, neglect, or significant life events.

            Social History
            - Education and work history.
            - Relationships and social support network.
            - Hobbies, interests, and activities.
            - Forensic history (if applicable).

            Substance Use History
            - Alcohol and drug use (current and past).
            - Tobacco and caffeine consumption.
            - Treatment or recovery history for substance use.

            Mental Status Examination (MSE)
            - Appearance, behavior, speech, and mood.
            - Thought processes and content.
            - Perceptions (e.g., hallucinations).
            - Cognition (e.g., memory, orientation, attention).
            - Insight and judgment.

            Functional Assessment
            - Activities of daily living (ADLs).
            - Employment and financial management.
            - Social and interpersonal functioning.

            Psychological Testing (if performed)
            - Summary of assessments (e.g., IQ tests, personality tests).
            - Key findings and interpretations.

            Diagnosis
            - Current DSM-5 or ICD-10 diagnoses.
            - Differential diagnoses (if applicable).

            Summary of Conflicting Information
            - Differences in patient, family, and clinical accounts.
            - Discrepancies in past records and current findings.

            Risk Assessment
            - Risk of self-harm or harm to others.
            - Vulnerabilities and protective factors.
            - Safety planning or crisis management strategies.

            Treatment Summary
            - Current treatment regimen (medications, therapies).
            - Treatment adherence and effectiveness.

            Recommendations
            - Short- and long-term treatment goals.
            - Suggestions for medications, therapies, or support services.
            - Community or NDIS supports (if applicable).

            Prognosis
            - Likely outcomes based on patient’s condition and engagement with treatment.
            ''',
            help="Customize the chatbot's personality by editing this template.", height=500
        )
        if st.session_state.get("pdf_processed") and st.session_state.api_authenticated:
            prompt = PROMPT_TEMPLATE
            with st.form("user_input_form"):
                user_question = st.text_input("Ask a question about your documents:")
                send_button = st.form_submit_button("Send")
            if send_button and user_question:
                handle_userinput(user_question, prompt)
        if not st.session_state.get("pdf_processed"): 
            st.caption("Please Upload Atleast 1 PDF Before Proceeding")
        if not st.session_state.api_authenticated:
            st.caption("Please Authenticate OpenAI API Before Proceeding")
    with code_tab:
        display_code()


if __name__ == '__main__':
    main()
