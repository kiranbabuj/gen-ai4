import streamlit as st
import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# Custom CSS to mimic Apple-like design with better input field visibility
st.markdown("""
    <style>
    /* Set global font family and base styles */
    html, body, [class*="css"]  {
        font-family: 'Helvetica Neue', sans-serif;
        background-color: #f8f8f8;
        color: #333333;
    }
    .stButton>button {
        background-color: #007aff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border: 1px solid #d1d1d1;
        padding: 12px;
        border-radius: 8px;
        background-color: white;
        color: black;  /* Set input text color to black */
    }
    .stFileUploader>div>div>button {
        background-color: #007aff;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px;
    }
    .stAlert {
        border-left: 4px solid #007aff;
    }
    .stDownloadButton>button {
        background-color: #007aff;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px;
    }
    .stSelectbox>div>div>div>div>div>button {
        background-color: white;
        color: #333;
        border: 1px solid #d1d1d1;
        padding: 10px;
        border-radius: 8px;
        font-size: 16px;
    }
    h1 {
        font-size: 40px;
        color: #333333;
        font-weight: bold;
    }
    h2, h3 {
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'question_asked' not in st.session_state:
    st.session_state.question_asked = False

# Streamlit UI for OpenAI API key input
st.title("Virtual Customer with Roles")

# Input API Key
api_key = st.text_input("Enter your OpenAI API key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API Key set successfully.")

    # Role selection for both assistant and user
    assistant_role = st.text_input("Enter the Assistant's Role", value="Customer Service Agent")
    user_role = st.text_input("Enter Your Role", value="Customer")

    # Template for prompting with roles
    prompt_template = f"""
    You are a {assistant_role}. Your job is to assist a {user_role} with their queries based on the provided document.
    Start by summarizing the document and asking a question as if you were interacting with a {user_role}.
    """

    # Display the selected roles
    st.write(f"**Assistant Role**: {assistant_role}")
    st.write(f"**User Role**: {user_role}")

    # Choose the LLM model
    model_choice = st.selectbox("Select an LLM model", ["gpt-4o-mini"])

    # Initialize the LLM
    st.write("Initializing LLM model...")
    llm = OpenAI(model=model_choice)

    # Embeddings section
    st.write("## Embeddings Section")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAI(model="gpt-4o-mini", max_tokens=300)

    # Document loading and querying
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded PDF locally
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load documents and create an index
        documents = SimpleDirectoryReader("./").load_data()
        index = VectorStoreIndex.from_documents(documents)

        # Initialize the query engine
        query_engine = index.as_query_engine()

        # Automatically query the document to initiate the conversation
        if not st.session_state.question_asked:
            st.write("## Assistant's Initial Conversation")
            # Assistant starts the conversation based on the PDF content
            initial_prompt = prompt_template + "\nProvide responses based on the roles selected and refer the document if needed."
            initial_response = query_engine.query(initial_prompt)
            # Add assistant's initial message to conversation history
            st.session_state.conversation_history.append(f"Assistant ({assistant_role}): {initial_response}")
            st.session_state.question_asked = True

            # Display the assistant's initial message
            st.write(f"Assistant: {initial_response}")

        # Allow the user to respond
        user_input = st.text_input(f"{user_role}'s Response:")

        if user_input:
            # Add user input to conversation history
            st.session_state.conversation_history.append(f"{user_role}: {user_input}")

            # Query the document for a response based on the user's input
            document_based_response = query_engine.query(user_input)

            # Safely convert document_based_response to string
            if isinstance(document_based_response, str):
                response_text = document_based_response
            else:
                response_text = str(document_based_response)

            # Calculate relevance score based on string similarity
            relevance_score = len(set(user_input.lower().split()) & set(response_text.lower().split())) * 10 // len(user_input.split())
            relevance_score = max(1, min(relevance_score, 10))  # Ensure the rating is between 1 and 10

            # Add assistant's response and follow-up question to conversation history
            st.session_state.conversation_history.append(f"Assistant ({assistant_role}): {response_text} (Rating: {relevance_score}/10)")
            st.session_state.question_asked = False  # Reset to allow another question

            # Display the assistant's response
            st.write(f"Assistant: {response_text} (Rating: {relevance_score}/10)")

            # Continue asking the next question
            follow_up_response = query_engine.query("Ask another relevant customer-like question.")
            st.session_state.conversation_history.append(f"Assistant ({assistant_role}): {follow_up_response}")
            st.write(f"Assistant: {follow_up_response}")

    # Display conversation history
    if st.session_state.conversation_history:
        st.write("## Conversation History")
        for entry in st.session_state.conversation_history:
            st.write(entry)

    # Download conversation history as a text file
    if st.session_state.conversation_history:
        conversation_text = "\n".join(st.session_state.conversation_history)
        st.download_button(
            label="Download Conversation History",
            data=conversation_text,
            file_name="conversation_history.txt",
            mime="text/plain",
        )
else:
    st.warning("Please enter your OpenAI API key to continue.")
