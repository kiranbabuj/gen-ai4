import streamlit as st
import os
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
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Prompt templates for roles
PROMPT_TEMPLATES = {
    "Bank Employee": "Your role is a professional banker, patiently answer all the questions by referring to the PDF document.",
    "Customer": "You are a new customer to the bank and sometimes you are an existing customer of Canara Bank. Your role is to know about bank products and services. You can show all kinds of emotions and behave like a natural customer randomly."
}

# Streamlit UI for OpenAI API key input
st.title("Conversational Assistant")

# Input API Key
api_key = st.text_input("Enter your OpenAI API key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.success("API Key set successfully.")

    # Select roles using dropdown menus
    st.session_state.user_role = st.selectbox(
        "Select user role:",
        options=["Customer", "Bank Employee"],
        index=0  # Default to "Customer"
    )
    
    st.session_state.assistant_role = st.selectbox(
        "Select assistant role:",
        options=["Bank Employee", "Customer"],
        index=0  # Default to "Bank Employee"
    )

    # Ensure roles are not the same
    if st.session_state.user_role == st.session_state.assistant_role:
        st.warning("User and Assistant roles cannot be the same. Please select different roles.")

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
            # Capture user input
            user_input = st.text_input("Your response:", key="user_input")  # Use session state key

            # If the user has entered a response
            if user_input:
                # Validate the user input and avoid empty inputs
                if user_input.strip() == "":
                    st.warning("Please enter a valid response.")
                else:
                    # Append user input to conversation history
                    st.session_state.conversation_history.append(f"{st.session_state.user_role}: {user_input}")

                    # Process the user input
                    document_based_response = query_engine.query(user_input)

                    response_text = str(document_based_response) if document_based_response else "No response generated."
                    
                    # If the user is a Bank Employee, generate feedback for their response
                    if st.session_state.user_role == "Bank Employee":
                        feedback_prompt = f"Evaluate the following response from a bank employee and suggest improvements: {user_input}"
                        
                        # Make sure to use the correct method to generate feedback
                        try:
                            feedback_response = llm.query(feedback_prompt)  # Use the appropriate method
                            st.session_state.conversation_history.append(f"Feedback: {feedback_response}")
                            st.write(f"Feedback: {feedback_response}")
                        except Exception as e:
                            st.error(f"Error generating feedback: {e}")
                    
                    # Append the assistant's response to history
                    st.session_state.conversation_history.append(f"{st.session_state.assistant_role}: {response_text}")

                    # Clear user input after processing
                    st.session_state.user_input = ""  # Clear the user input field

        with col2:
            if st.button("Clear History"):
                # Clear conversation history and reset flags
                st.session_state.conversation_history = []
                st.session_state.question_asked = False
                st.session_state.user_input = ""  # Clear the user input field

        # Automatically query the document to initiate the conversation
        if not st.session_state.question_asked:
            st.write("## Assistant's Initial Conversation")

            # Generate role-specific initial response
            if st.session_state.assistant_role == "Bank Employee":
                initial_response = "Good morning, welcome to Canara Bank. How can I assist you today?"
            elif st.session_state.assistant_role == "Customer":
                initial_response = "Hi there! I'm a customer, looking for assistance."

            # Add assistant's initial message to conversation history
            st.session_state.conversation_history.append(f"{st.session_state.assistant_role}: {initial_response}")
            st.session_state.question_asked = True

            # Display the assistant's initial message
            st.write(f"{st.session_state.assistant_role}: {initial_response}")

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
    st.warning("Please enter your OpenAI API key.")
