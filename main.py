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
from htmlTemplates import css, bot_template, user_template
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
        return_source_documents=False,  # Disable source docs here
        output_key="answer"
    )
    return conversation_chain


def handle_userinput(user_question, prompt):
    # Run the chain, now it will only return 'answer'
    chain_output = st.session_state.conversation({'question': (prompt + user_question)})
    answer = chain_output['answer']
    
    # Retrieve documents separately
    source_docs = st.session_state.conversation.retriever.get_relevant_documents(user_question)

    st.session_state.chat_history = chain_output['chat_history']
    with st.spinner('Generating response...'):
        display_convo_with_source(prompt, answer, source_docs)


def display_convo_with_source(prompt, answer, source_documents):
    # Display the bot's response
    st.markdown(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

    # Display source documents
    if source_documents:
        st.markdown("### Source Documents:")
        for idx, doc in enumerate(source_documents, start=1):
            st.markdown(f"**Source {idx}:** {doc.page_content}")
    else:
        st.markdown("**No specific documents identified.**")

    # Display the user's question
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 != 0:  # User's input
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
    global llm
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
        MODEL = st.selectbox(label='Model', options=['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo-0125'])
        TEMP = st.slider("Temperature", 0.0, 1.0, 0.0, help="Temperature for the model to generate responses. Lower values are more deterministic, higher values are more creative.")
    


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
    deploy_tab, transcribe= st.tabs(["Note Synthesis", "Ambient AI Note Generation"])
    with deploy_tab:
        st.title(":books: Medical-Document ChatBot 🤖")
        sidebar()
        PROMPT_TEMPLATE = st.text_area(
            label="Prompt Template", 
            value='''You are a precise, respectful, and reliable assistant dedicated to delivering accurate and detailed responses based strictly on the provided context. If any information is unclear or unsupported, explicitly state that clarification is needed. Responses must always be comprehensive, well-structured, and tailored to the nature of the question, with a focus on clarity, accuracy, and relevance.

When answering, ensure that no critical information is missed or overlooked. Your responses should be exhaustive, addressing every aspect of the query or context provided. 

Guidelines for Responses:
1. **Comprehensive Coverage**: Ensure that all relevant details are included in your response. Do not omit any information unless explicitly directed to do so.
2. **Clarity and Structure**: Use clear and logical formatting to ensure readability and ease of understanding. Bullet points, numbered lists, and headings/subheadings should be used where appropriate.
3. **Addressing Ambiguity**: If any details are missing or unclear, state explicitly what further information is required to provide a complete response.
4. **Accuracy and Precision**: Ensure that all information provided is accurate, avoiding unsupported assumptions. If external context is required and unavailable, specify the limitations in the response.

---

**General Information Response Format**

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
- Description of primary concerns or issues leading to the assessment or treatment.
- Identification of the referral source (e.g., self, family, GP, court).

Presenting Complaints
- Detailed account of the patient’s current symptoms, concerns, or challenges.
- Context or triggers for recent episodes (if applicable).
- Duration and progression of symptoms.

Psychiatric History
- Previous diagnoses and treatment history.
- Record of psychiatric hospitalizations (if any).
- Detailed accounts of episodes of mania, depression, anxiety, psychosis, etc.
- History of suicidal ideation or attempts, including specific incidents.
- Overview of past medications, adherence, and outcomes.

Medical History
- Comprehensive record of past and current medical conditions.
- Details of medications, both psychiatric and non-psychiatric.
- Allergies and sensitivities.
- Relevant lab or diagnostic findings.

Developmental History
- Information about prenatal, perinatal, and early childhood development.
- Milestones in physical, social, emotional, and cognitive growth.
- Academic and social history during childhood and adolescence.

Family History
- Psychiatric and medical conditions in close and extended family members.
- Dynamics of family relationships and history of significant life events (e.g., abuse, neglect, trauma).

Social History
- Overview of educational and occupational history.
- Details about interpersonal relationships and social support.
- Information about hobbies, interests, and personal pursuits.
- Record of forensic or legal issues, if relevant.

Substance Use History
- Comprehensive account of alcohol, drug, and substance use (current and past).
- Details on tobacco and caffeine consumption.
- History of substance use treatment or recovery efforts.

Mental Status Examination (MSE)
- Description of the patient’s appearance, behavior, and speech.
- Detailed observation of mood, affect, and thought processes.
- Report on perceptions (e.g., hallucinations), cognition (memory, orientation, attention), and insight/judgment.

Functional Assessment
- Evaluation of daily living skills and routines.
- Capacity to maintain employment and manage finances.
- Social and interpersonal functioning.

Psychological Testing (if applicable)
- Summary of testing methods used (e.g., IQ, personality assessments).
- Key findings and interpretations of results.

Diagnosis
- Current DSM-5 or ICD-10 diagnoses.
- Consideration of differential diagnoses, if applicable.

Treatment Summary
- Description of the current treatment plan (e.g., medications, therapies).
- Assessment of treatment adherence and effectiveness.

Recommendations
- Clear, actionable suggestions for short- and long-term treatment goals.
- Recommendations for medications, therapy modalities, or support services.
- Additional supports or referrals (e.g., community services, NDIS, housing assistance).

Prognosis
- Likely outcomes based on current condition, treatment adherence, and support system.
- Discussion of factors that may influence recovery.

---

When addressing **conflicts in medical/psychiatric history** and **discrepancies in medications**, ensure a detailed and systematic approach. Clearly identify, explain, and analyze all conflicts or inconsistencies. No detail should be missed.

**Conflicts and Discrepancies in Medical/Psychiatric History**
1. **Condition/Diagnosis Name**:
   - **Initial diagnosis**: Provide details on the diagnosis, age at onset, and any reported features or symptoms as documented in earlier records.
   - **Additional conflicts**: Include discrepancies in patient, family, or clinical accounts, including differences in how symptoms or behaviors are described (e.g., cultural perceptions, family dynamics).
   - **Impact**: Explain the implications of these discrepancies on diagnosis, treatment planning, medication choices, or continuity of care.

(Repeat for each condition/diagnosis as needed. Ensure all relevant conditions are included.)

---

**Discrepancies in Medication History**
1. **Medication Name**:
   - **Prescribed usage**: Detail the medication’s purpose, prescribed dosage, and intended treatment timeline.
   - **Conflicting reports**: Highlight variations between self-reported usage, adherence, or side effects and documented records.
   - **Effectiveness and adherence**: Discuss inconsistencies in perceived effectiveness, adherence patterns, or reported side effects across sources (e.g., patient, family, clinical documentation).
   - **Impact**: Explain how these discrepancies might influence current or future treatment safety, efficacy, or the trustworthiness of self-reports.

(Repeat for each medication as needed. Include every identified discrepancy in the medication history.)

---

**Example Output Format (Expanded for Multiple Entries)**

**Conflicts and Discrepancies in Medical/Psychiatric History**
1. **Condition A (e.g., ADHD)**:
   - **Initial diagnosis**: ADHD, combined type, diagnosed at age 12, with treatment initially starting with stimulant medications such as dexamphetamine. Reports from early records note hyperactivity and poor focus in school.
   - **Recent records**: Describe ADHD as predominantly inattentive type, with no hyperactivity mentioned in recent consultations.
   - **Symptom management conflicts**: Patient denies past stimulant usage due to perceived addiction risks, while earlier records document regular adherence until age 18.
   - **Additional conflicts**: Family describes symptoms as "minimal" and doubts the initial diagnosis, while school reports note significant academic struggles due to attention deficits.
   - **Impact**: These discrepancies could lead to uncertainty in choosing effective ADHD medications or understanding the historical impact of the condition on functioning.

2. **Condition B (e.g., Bipolar Disorder)**:
   - **Reported onset**: Diagnosed with Bipolar II Disorder at age 28 after a manic episode involving impulsive spending and sleep disturbances.
   - **Symptom management conflicts**: Patient denies current or past mood stabilization medications (e.g., Lithium), while clinical records confirm long-term usage with noted effectiveness.
   - **Additional conflicts**: Family reports differing accounts of manic behaviors, describing them either as "creative bursts" or as pathological impulsivity. This variability affects family dynamics and perceptions of treatment adherence.
   - **Impact**: Conflicting narratives on symptom severity and adherence to mood stabilizers complicate the development of a consistent treatment plan.

3. **Condition C (e.g., Substance Use)**:
   - **Reported abstinence**: Patient reports alcohol use disorder in remission since age 30, with no further substance use.
   - **Conflicting reports**: Recent records from a hospitalization indicate cannabis use during a manic phase, contradicting claims of full abstinence.
   - **Impact**: This discrepancy raises concerns about self-awareness, honesty in self-reporting, and potential ongoing substance use impacting mood stability.

---

**Discrepancies in Medication History**
1. **Medication A (e.g., Lithium)**:
   - **Prescribed usage**: Frequently prescribed for mood stabilization, with documented efficacy in reducing manic and depressive episodes.
   - **Conflicting reports**: Patient denies ever using Lithium, despite clinical records showing adherence issues due to side effects like tremors and nausea.
   - **Effectiveness and adherence**: Clinical notes confirm benefit during periods of adherence, but patient’s denial of use creates challenges in maintaining continuity of care.
   - **Impact**: This raises concerns about potential medication non-adherence in future treatments and the reliability of patient-reported history.

2. **Medication B (e.g., Venlafaxine)**:
   - **Prescribed usage**: Titrated to 150 mg for managing depressive symptoms during a recent inpatient admission.
   - **Conflicting reports**: Patient denies its use, claiming “self-management” of depressive symptoms without medication.
   - **Effectiveness and adherence**: Records indicate symptom improvement with Venlafaxine, but self-reports contradict this, risking potential relapse if not continued.
   - **Impact**: The discrepancy challenges accurate planning for ongoing management of depression.

3. **Medication C (e.g., Metformin)**:
   - **Prescribed usage**: Recently started for managing type 2 diabetes, with semaglutide planned for additional weight control.
   - **Conflicting reports**: Patient mentions willingness to take diabetes medication but displays inconsistent adherence as per pharmacy records.
   - **Impact**: Poor adherence risks worsening glycemic control and complications, necessitating stricter monitoring and education.

4. **Medication D (e.g., Quetiapine)**:
   - **Prescribed usage**: Occasionally prescribed as a sleep aid during hospitalizations, with documented side effects like excessive sedation.
   - **Conflicting reports**: Patient denies ever using Quetiapine, despite discharge summaries documenting its use for sleep disturbances.
   - **Impact**: Misreporting medication history limits the clinician’s ability to assess its role and benefits for managing sleep or mood disorders.

(Repeat this structure for all identified medications.)

---

Use this detailed structure to ensure all responses are exhaustive, well-organized, and address every aspect of the question or issue raised. Always prioritize clarity, precision, and relevance in your answers.
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
    with transcribe:
        from openai import OpenAI
        client = OpenAI()

        # Set up the Streamlit interface
        st.title("MP3 File Transcription App")

        # Allow the user to upload an MP3 file
        uploaded_file1 = st.file_uploader("Upload an MP3 file", type=["mp3"])

        if uploaded_file1 is not None:
            try:
                # Display a message to the user
                st.info("Transcribing your audio file, please wait...")

                # Perform transcription using Whisper API
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=uploaded_file1
                )

                # Display the transcription
                st.subheader("Transcription")
                st.text(transcription.text)

            except Exception as e:
                # Handle any errors that occur
                st.error(f"An error occurred: {e}")
        else:
            st.write("Please upload an MP3 file to begin.")



if __name__ == '__main__':
    main()
