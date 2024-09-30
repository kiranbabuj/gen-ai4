import streamlit as st
import os
import pandas as pd
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader


class Role:
    """Base class for roles."""
    def __init__(self, role_name):
        self.role_name = role_name

    def respond(self, user_input, query_engine):
        """Method to respond based on user input."""
        raise NotImplementedError("Subclasses should implement this!")


class Customer(Role):
    """Customer role implementation."""
    def __init__(self):
        super().__init__("Customer")

    def respond(self, user_input, query_engine):
        """Generate a response from the LLM."""
        llm = OpenAI(model="gpt-4o-mini")  # Choose your model
        messages = [
            ChatMessage(role="system", content="You are a customer looking for assistance."),
            ChatMessage(role="user", content=user_input),
        ]
        response = llm.chat(messages)
        return response


class BankEmployee(Role):
    """Bank Employee role implementation."""
    def __init__(self):
        super().__init__("Bank Employee")

    def respond(self, user_input, query_engine):
        """Generate a document-based response."""
        document_based_response = query_engine.query(user_input)
        feedback_text = evaluate_response(user_input, document_based_response)
        return document_based_response, feedback_text


def evaluate_response(user_input, response_text):
    """Evaluate the response based on the user input and provide feedback."""
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
role_choice = st.selectbox("Select your role:", options=["Customer", "Bank Employee"])

# Create role instances based on user choice
if role_choice == "Customer":
    current_role = Customer()
else:
    current_role = BankEmployee()

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

    # Process user input if available
    if user_input:
        # Add user input to conversation history
        st.session_state.conversation_history.append(f"{current_role.role_name}: {user_input}")

        # Get the response based on the role
        if isinstance(current_role, BankEmployee):
            response_text, feedback_text = current_role.respond(user_input, query_engine)
            st.session_state.conversation_history.append(f"{current_role.role_name}: {response_text} (Feedback: {feedback_text})")
        else:
            response_text = current_role.respond(user_input, query_engine)
            st.session_state.conversation_history.append(f"{current_role.role_name}: {response_text}")

        st.session_state.question_asked = False  # Reset to allow another question

        # Display the assistant's response
        st.write(f"{current_role.role_name}: {response_text}")

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

            if role == "Customer":
                # If the role is the same as the last one, append to the last message
                if last_role == "Customer":
                    customer_messages[-1] += f" | {message.strip()}"
                else:
                    customer_messages.append(message.strip())
                    bank_employee_messages.append("")  # Empty entry for the bank employee column
            else:
                # If the role is the same as the last one, append to the last message
                if last_role == "Bank Employee":
                    bank_employee_messages[-1] += f" | {message.strip()}"
                else:
                    bank_employee_messages.append(message.strip())
                    customer_messages.append("")  # Empty entry for the customer column

            # Update the last role
            last_role = role

        # Create a DataFrame
        conversation_df = pd.DataFrame({
            "Customer": customer_messages,
            "Bank Employee": bank_employee_messages
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
