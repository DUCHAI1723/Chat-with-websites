# Import libraries
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# Load environment variables
load_dotenv()

# Define helper functions
def get_vectorstore_from_url(url, persist_directory="./db"): # lấy dữ liệu từ website -> chia chunks -> embeddings
    try:
        # Get the text in document form
        loader = WebBaseLoader(url)
        document = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(document)

        # Create a vectorstore from the chunks
        vector_store = Chroma.from_documents(
            document_chunks, OpenAIEmbeddings(), persist_directory=persist_directory
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()  # khởi tạo đối tượng mô hình LLM, mô hình này xử lý các yêu cầu, hiểu lịch sử hội thoại và tạo các truy vấn tìm kiếm
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI() 

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store) # truy xuất thông tin và ngữ cảnh
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)  # dựa vào thông tin đã truy xuất và ngữ cảnh để tạo câu trả lời

    try:
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        if not response or not response.get("answer") or not response.get("context"):
            return "I'm sorry, I couldn't find any relevant information from the provided website."
        return response['answer']
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I couldn't process that."

# App config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if not website_url:
    st.info("Please enter a website URL")
else:
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    if st.session_state.vector_store:
        # User input
        user_query = st.chat_input("Type your message here...")
        if user_query:
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        # Conversation display
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    else:
        st.error("Failed to initialize vector store. Please check the URL and try again.")
