# Medical-Chatbot

---

#### **Project Overview**
The Medical-Document ChatBot is a Streamlit-based web application designed to allow users to upload multiple PDF documents and interact with the content via a conversational chatbot. It uses the OpenAI API for language generation, LangChain for conversational retrieval, and FAISS for vector storage, enabling intelligent querying and analytics of uploaded documents.

---

#### **Features**
1. **PDF Document Analysis**:
   - Upload multiple PDF files.
   - Automatically processes text from the documents and chunks it into manageable pieces for analysis.
2. **Conversational Chatbot**:
   - Interacts using OpenAI models (e.g., GPT-4 or GPT-3.5).
   - Uses a customizable prompt template for user-defined chatbot behavior.
3. **Chat History**:
   - Maintains a session-based conversation history.
4. **Visualizations**:
   - Integrates with custom visual analytics for summarizing document insights.
5. **Authentication**:
   - Secure OpenAI API key authentication through environment variables or user input.

---

#### **Installation Instructions**
1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Install Dependencies**:
   Make sure you have Python 3.8+ installed. Use the following command to install dependencies:
   ```
   pip install -r requirements.txt
   ```
   The dependencies include:
   - Streamlit: Web interface framework.
   - OpenAI: Language model interaction.
   - FAISS: Vector database for efficient text retrieval.
   - LangChain: Conversational AI toolkit.
   - PyPDF2: PDF parsing.
   - Plotly: Data visualization.

3. **Set Up Environment Variables**:
   - Create a `.env` file to securely store your OpenAI API key:
     ```
     OPENAI_API_KEY=<your-api-key>
     ```
   - Alternatively, the app allows direct input of the API key.

4. **Run the Application**:
   Start the Streamlit server:
   ```
   streamlit run main.py
   ```

---

#### **How It Works**
1. **Initialization**:
   - `init_ses_states()`: Initializes session states, ensuring a clean start for every session.

2. **Document Processing**:
   - `get_pdfs_text()`: Reads and extracts text from uploaded PDFs.
   - `get_text_chunks()`: Splits raw text into chunks suitable for vector embedding.
   - `get_vectorstore()`: Creates a FAISS-based vector store for document indexing.

3. **Conversation Handling**:
   - `get_conversation_chain()`: Establishes a conversational retrieval chain with memory and context awareness.
   - `handle_userinput()`: Handles user questions and generates appropriate responses using the conversation chain.

4. **Authentication**:
   - `OpenAIAuthenticator`: Verifies API key validity before enabling access to chatbot functionalities.

5. **Chatbot Settings**:
   - Customize model (`GPT-4` or `GPT-3.5`) and temperature (creativity of responses).
   - Define prompt templates for personalized chatbot behavior.

---

#### **Application Structure**
1. **Main Application (`main.py`)**:
   - Core logic and Streamlit interface.
   - Sidebar for API key input, chatbot settings, and document upload.
2. **Custom Modules**:
   - `htmlTemplates`: HTML templates for chatbot conversation formatting.
   - `codeDisplay`: Utility to display Python code within the Streamlit interface.
   - `textFunctions`: Functions for text extraction and processing.
   - `vizFunctions`: Utilities for visual analytics.

---

#### **Usage Guide**
1. Launch the application using `streamlit run main.py`.
2. **Authenticate**:
   - Input your OpenAI API key in the sidebar under "OpenAI API Authentication".
3. **Upload PDFs**:
   - Drag and drop PDFs into the "Your Documents" section.
4. **Process Files**:
   - Click "Process Files + New Chat" to prepare the documents for analysis.
5. **Ask Questions**:
   - Type questions in the provided input box to interact with document content.

---

#### **Customizations**
1. **Modify Prompt Template**:
   - Change the default prompt in the "Prompt Template" section.
   - Adjust formatting, tone, or guidelines to match specific requirements.
2. **Add Custom Visualizations**:
   - Implement additional analytics in the `vizFunctions` module.

---

#### **Troubleshooting**
1. **Missing Dependencies**:
   - Ensure all dependencies listed in `requirements.txt` are installed.
2. **API Key Errors**:
   - Verify the validity of your OpenAI API key and check `.env` setup.
3. **Document Processing Issues**:
   - Ensure uploaded PDFs are valid and properly formatted.

---

#### **Future Enhancements**
- Add support for non-PDF document formats (e.g., Word, Excel).
- Enhance visualization options.
- Improve memory management for larger documents.

---

For further inquiries or contributions, feel free to raise an issue or submit a pull request to the repository.
