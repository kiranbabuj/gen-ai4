import streamlit as st
import os
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# Initialize conversation history and state in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'assistant_followup_question' not in st.session_state:
    st.session_state.assistant_followup_question = None  # To store assistant's follow-up question
if 'awaiting_user_response' not in st.session_state:
    st.session_state.awaiting_user_response = False  # Control flow for question-response cycle
if 'question_cycle_active' not in st.session_state:
    st.session_state.question_cycle_active = False  # To track if question cycle is ongoing

# Streamlit UI for OpenAI API key input
st.title("Virtual Customer - Training Assistant")

# Input API Key
api_key = st.text_input("Enter your OpenAI API key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API Key set successfully.")

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

        # Initialize the assistant's conversation based on the PDF content
        query_engine = index.as_query_engine()

        if not st.session_state.question_cycle_active:
            # Assistant starts the conversation based on the PDF content
            initial_response = query_engine.query(
                "You are a customer of the bank. Ask relevant single-line questions as a customer would ask, by summarizing the document. The user will give the answer accordingly, and you will rate the user on a scale of 1 to 10 based on how accurate the answer is. Continue asking questions after the user provides an answer."
            )

            # Add assistant's initial message to conversation history
            st.session_state.conversation_history.append(f"Assistant: {initial_response}")
            st.session_state.assistant_followup_question = initial_response
            st.session_state.awaiting_user_response = True  # Start expecting the user's input
            st.session_state.question_cycle_active = True  # Begin automatic question cycle

        # Display the assistant's current question at the top
        st.markdown(f"<h4 style='color: blue;'>Assistant: {st.session_state.assistant_followup_question}</h4>", unsafe_allow_html=True)

        # Allow the user to respond
        user_input = st.text_input("Your response:")

        if user_input and st.session_state.awaiting_user_response:
            # Add user input to conversation history
            st.session_state.conversation_history.append(f"User: {user_input}")

            # Use user input to query the document for relevance
            document_based_response = query_engine.query(user_input)

            # Calculate a mock relevance score based on string similarity or token matching (this is a simple placeholder)
            relevance_score = len(set(user_input.lower().split()) & set(document_based_response.lower().split())) * 10 // len(user_input.split())
            relevance_score = max(1, min(relevance_score, 10))  # Ensure the rating is between 1 and 10

            # Generate the assistant's rating and follow-up question
            followup_question = query_engine.query("Ask the next relevant question.")

            # Add assistant's response and follow-up question to conversation history
            st.session_state.conversation_history.append(f"Assistant: {document_based_response} (Rating: {relevance_score}/10)")
            st.session_state.conversation_history.append(f"Assistant: {followup_question}")

            # Update the follow-up question and continue the loop
            st.session_state.assistant_followup_question = followup_question
            st.session_state.awaiting_user_response = True  # Continue expecting user input

        # Display conversation history (latest messages on top)
        if st.session_state.conversation_history:
            st.write("## Conversation History")
            for entry in reversed(st.session_state.conversation_history):
                if entry.startswith("Assistant:"):
                    st.markdown(f"<div style='color: blue; background-color: #e6f7ff; padding: 8px; border-radius: 8px;'>{entry}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color: green; background-color: #e8ffe8; padding: 8px; border-radius: 8px;'>{entry}</div>", unsafe_allow_html=True)

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
