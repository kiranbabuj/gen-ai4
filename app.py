import streamlit as st
import os
import random
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# Initialize conversation history and roles in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'question_asked' not in st.session_state:
    st.session_state.question_asked = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = "Customer"
if 'assistant_role' not in st.session_state:
    st.session_state.assistant_role = "Bank Employee"

# Define improved prompt templates for roles
prompt_templates = {
    "Bank Employee": (
        "You are a professional and knowledgeable bank employee at Canara Bank. "
        "You provide detailed, accurate, and patient responses to customer questions. "
        "Always refer to relevant sections of the provided PDF document when answering queries about bank products, services, or policies."
    ),
    "Customer": (
        "You are a new or existing customer of Canara Bank. You ask questions about the bank's products and services. "
        "Your emotions and tone may vary—ranging from curiosity to frustration—depending on the context of your interaction."
    )
}

# Streamlit UI for OpenAI API key input
st.title("Conversational Assistant")

# Input API Key
api_key = st.text_input("Enter your OpenAI API key", type="password")

# Store API key in session state
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API Key set successfully.")

    # Select user role using a dropdown menu
    st.session_state.user_role = st.selectbox(
        "Select your role:",
        options=["Customer", "Bank Employee"],
        index=["Customer", "Bank Employee"].index(st.session_state.user_role)  # Set to current user role
    )

    # Set assistant role based on the selected user role
    st.session_state.assistant_role = "Bank Employee" if st.session_state.user_role == "Customer" else "Customer"

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

        # Layout for user input and clear history button side by side
        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_input("Your response:")

        with col2:
            if st.button("Clear History"):
                # Clear conversation history and reset flags except for the API key
                st.session_state.conversation_history = []
                st.session_state.question_asked = False
                st.session_state.user_role = "Customer"  # Reset to default role
                st.session_state.assistant_role = "Bank Employee"  # Reset to default role

        # Automatically initiate the conversation with initial role-based prompt
        if not st.session_state.question_asked:
            st.write("## Assistant's Initial Conversation")

            # Generate role-specific initial response using prompt templates
            if st.session_state.assistant_role == "Bank Employee":
                initial_response = "Good morning! Welcome to Canara Bank. How can I assist you today with our banking products or services?"
            else:
                initial_response = "Hi! I’m looking to understand more about Canara Bank's savings account options. Can you help?"

            # Add assistant's initial message to conversation history
            st.session_state.conversation_history.append(f"{st.session_state.assistant_role}: {initial_response}")
            st.session_state.question_asked = True

            # Display the assistant's initial message
            st.write(f"{st.session_state.assistant_role}: {initial_response}")

        # Process user input if available
        if user_input:
            # Add user input to conversation history
            st.session_state.conversation_history.append(f"{st.session_state.user_role}: {user_input}")

            # Query the document for a response based on the user's input
            document_based_response = query_engine.query(
                f"Based on the customer input: {user_input}, check if relevant sections of the document exist. "
                "If yes, provide a detailed response. If no, respond based on your expertise."
            )

            # Safely convert document_based_response to string
            response_text = str(document_based_response) if document_based_response else "Sorry, I couldn't find relevant information."

            # Calculate a relevance score based on string similarity
            relevance_score = len(set(user_input.lower().split()) & set(response_text.lower().split())) * 10 // len(user_input.split())
            relevance_score = max(1, min(relevance_score, 10))  # Ensure the rating is between 1 and 10

            # Add assistant's response and follow-up question to conversation history
            st.session_state.conversation_history.append(f"{st.session_state.assistant_role}: {response_text} (Rating: {relevance_score}/10)")
            st.session_state.question_asked = False  # Reset to allow another question

            # Display the assistant's response
            st.write(f"{st.session_state.assistant_role}: {response_text} (Rating: {relevance_score}/10)")

            # Generate role-specific follow-up response using prompt templates
            follow_up_response = ""
            if st.session_state.assistant_role == "Bank Employee":
                follow_up_response = query_engine.query("As a bank employee, ask the customer another banking-related question.")
            elif st.session_state.assistant_role == "Customer":
                emotions = ["curious", "frustrated", "confused"]
                emotion = random.choice(emotions)
                follow_up_response = query_engine.query(f"As a {emotion} customer, ask a follow-up question.")

            # Add follow-up response to the conversation history
            st.session_state.conversation_history.append(f"{st.session_state.assistant_role}: {follow_up_response}")
            st.write(f"{st.session_state.assistant_role}: {follow_up_response}")

    # Display conversation history with different colors and numbering
    if st.session_state.conversation_history:
        st.write("## Conversation History")
        for idx, entry in enumerate(st.session_state.conversation_history, start=1):
            if entry.startswith(st.session_state.user_role):
                st.markdown(f"<div style='color:blue;'>**{idx}. {entry}**</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color:green;'>**{idx}. {entry}**</div>", unsafe_allow_html=True)

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
