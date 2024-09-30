import streamlit as st
import os
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'question_asked' not in st.session_state:
    st.session_state.question_asked = False

# Streamlit UI for OpenAI API key input
st.title("Virtual Assistant with Role-based Conversation")

# Input API Key
api_key = st.text_input("Enter your OpenAI API key", type="password")

# Input for the assistant role
assistant_role = st.text_input("Enter the role for your assistant (e.g., 'Bank Customer', 'Support Agent'): ", value="Assistant")

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

        # Initialize the query engine
        query_engine = index.as_query_engine()

        # Automatically query the document to initiate the conversation
        if not st.session_state.question_asked:
            st.write(f"## {assistant_role}'s Initial Conversation")
            # Assistant starts the conversation based on the PDF content
            initial_response = query_engine.query("Summarize the document and ask a customer-like question.")
            st.session_state.conversation_history.append({"role": assistant_role, "message": initial_response})
            st.session_state.question_asked = True

            # Display the assistant's initial message
            st.write(f"{assistant_role}: {initial_response}")

        # Allow the user to respond
        user_input = st.text_input("Your response:")

        if user_input:
            # Add user input to conversation history
            st.session_state.conversation_history.append({"role": "User", "message": user_input})

            # Query the document for a response based on the user's input
            document_based_response = query_engine.query(user_input)

            # Safely convert document_based_response to string
            response_text = str(document_based_response) if isinstance(document_based_response, str) else document_based_response

            # Calculate relevance score based on string similarity
            relevance_score = len(set(user_input.lower().split()) & set(response_text.lower().split())) * 10 // len(user_input.split())
            relevance_score = max(1, min(relevance_score, 10))  # Ensure the rating is between 1 and 10

            # Add assistant's response and follow-up question to conversation history
            st.session_state.conversation_history.append({"role": assistant_role, "message": f"{response_text} (Rating: {relevance_score}/10)"})
            st.session_state.question_asked = False  # Reset to allow another question

            # Display the assistant's response
            st.write(f"{assistant_role}: {response_text} (Rating: {relevance_score}/10)")

            # Continue asking the next question
            follow_up_response = query_engine.query("Ask another relevant customer-like question.")
            st.session_state.conversation_history.append({"role": assistant_role, "message": follow_up_response})
            st.write(f"{assistant_role}: {follow_up_response}")

    # Clear history button
    if st.button("Clear Conversation History"):
        st.session_state.conversation_history = []

    # Display conversation history with coloring and numbering
    if st.session_state.conversation_history:
        st.write("## Conversation History")
        for i, entry in enumerate(st.session_state.conversation_history):
            role_color = "#007bff" if entry["role"] == assistant_role else "#28a745"
            st.markdown(f"<p style='color: {role_color};'><strong>{i + 1}. {entry['role']}:</strong> {entry['message']}</p>", unsafe_allow_html=True)

    # Download conversation history as a text file
    if st.session_state.conversation_history:
        conversation_text = "\n".join([f"{i + 1}. {entry['role']}: {entry['message']}" for i, entry in enumerate(st.session_state.conversation_history)])
        st.download_button(
            label="Download Conversation History",
            data=conversation_text,
            file_name="conversation_history.txt",
            mime="text/plain",
        )
else:
    st.warning("Please enter your OpenAI API key to continue.")
