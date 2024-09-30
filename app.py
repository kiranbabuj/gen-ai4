import streamlit as st
import os
import pandas as pd
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# Define the evaluate_response function before using it
def evaluate_response(user_input, response_text):
    """
    Evaluate the response based on the user input and provide feedback.
    """
    # For training, we will check if the response includes key terms.
    key_terms = ['bank', 'loan', 'account', 'deposit', 'withdrawal', 'interest rate']
    missing_terms = [term for term in key_terms if term not in response_text.lower()]

    if missing_terms:
        feedback = f"Missing key terms: {', '.join(missing_terms)}. Consider including them."
    else:
        feedback = "Great response! It covers all necessary points."

    return feedback

# Initialize conversation history and roles in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'question_asked' not in st.session_state:
    st.session_state.question_asked = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = "Customer"
if 'assistant_role' not in st.session_state:
    st.session_state.assistant_role = "Bank Employee"
if 'api_key' not in st.session_state:
    st.session_state.api_key = None  # Initialize API key in session state

# Streamlit UI for OpenAI API key input
st.title("Conversational Assistant")

# Input API Key (store in session state)
if st.session_state.api_key is None:  # Only show input if the API key is not set
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    if api_key:
        st.session_state.api_key = api_key  # Store API key in session state
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API Key set successfully.")
else:
    st.success("API Key is already set.")

# Select roles using a dropdown menu
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
            # Clear conversation history and reset flags
            st.session_state.conversation_history = []
            st.session_state.question_asked = False

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

    # Process user input if available
    if user_input:
        # Add user input to conversation history
        st.session_state.conversation_history.append(f"{st.session_state.user_role}: {user_input}")

        # Check if the user is a Bank Employee
        if st.session_state.user_role == "Bank Employee":
            # The Bank Employee will evaluate the response based on the document
            document_based_response = query_engine.query(user_input)
            response_text = document_based_response if isinstance(document_based_response, str) else str(document_based_response)

            # Evaluate the response for improvement
            feedback_text = evaluate_response(user_input, response_text)
            st.session_state.conversation_history.append(f"{st.session_state.assistant_role}: {response_text} (Feedback: {feedback_text})")
        
        else:
            # Generate LLM response for other roles
            response_text = llm.chat([
                ChatMessage(role="system", content="You are a finance domain expert."),
                ChatMessage(role="user", content=user_input),
            ])
            st.session_state.conversation_history.append(f"{st.session_state.assistant_role}: {response_text}")

        st.session_state.question_asked = False  # Reset to allow another question

        # Display the assistant's response
        st.write(f"{st.session_state.assistant_role}: {response_text}")

    # Display the follow-up response here after embeddings section
    if st.session_state.question_asked:
        # Generate role-specific follow-up response
        if st.session_state.assistant_role == "Bank Employee":
            follow_up_response = query_engine.query("As a bank employee, ask the customer another banking-related question.")
        elif st.session_state.assistant_role == "Customer":
            follow_up_response = query_engine.query("As a customer, ask a follow-up question to the assistant.")

        # Add follow-up response to the conversation history
        st.session_state.conversation_history.append(f"{st.session_state.assistant_role}: {follow_up_response}")
        st.write(f"{st.session_state.assistant_role}: {follow_up_response}")

    # Display conversation history in a side-by-side format
    if st.session_state.conversation_history:
        st.write("## Conversation History")

        # Prepare data for the DataFrame
        customer_messages = []
        bank_employee_messages = []

        # Track the last user role
        last_role = None
        
        for entry in st.session_state.conversation_history:
            role, message = entry.split(": ", 1)

            if role == st.session_state.user_role:
                # If the role is the same as the last one, append to the last message
                if last_role == st.session_state.user_role:
                    customer_messages[-1] += f" | {message.strip()}"
                else:
                    customer_messages.append(message.strip())
                    bank_employee_messages.append("")  # Empty entry for the bank employee column
            else:
                # If the role is the same as the last one, append to the last message
                if last_role == st.session_state.assistant_role:
                    bank_employee_messages[-1] += f" | {message.strip()}"
                else:
                    bank_employee_messages.append(message.strip())
                    customer_messages.append("")  # Empty entry for the customer column

            # Update the last role
            last_role = role

        # Create a DataFrame
        conversation_df = pd.DataFrame({
            st.session_state.user_role: customer_messages,
            st.session_state.assistant_role: bank_employee_messages
        })

        # Display the DataFrame as a table
        st.table(conversation_df)

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
