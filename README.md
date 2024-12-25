
# Chat with a Website using LangChain and Streamlit 

Welcome to this guide for building a LangChain-powered chatbot with a Streamlit interface! This project will show you how to create a chatbot capable of interacting with websites, extracting relevant information, and providing responses through a user-friendly GUI

# Features
1. Website Data Interaction: Extracts and interacts with information from web pages using LangChain.
2. Large Language Model Support: Compatible with GPT-4, Llama2, Mistral, Ollama, and others. Default configuration uses GPT-4, but you can switch to a model of your choice
3. Intuitive Streamlit Interface: Provides an easy-to-use graphical interface suitable for all users
4. Python Implementation: Fully written in Python for simplicity and flexibility.
   
# Understanding Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) enriches an LLM’s knowledge by incorporating additional context. Here’s how it works:
1. Text Vectorization: All relevant text is vectorized.
2. Similarity Search: The system identifies text segments most relevant to the query.
3. Enhanced Prompting: This context is added to the model prompt to improve response quality.
Below is a diagram that shows the process: 
![example](/Images/HTML-rag-diagram.jpg)

# Setup Instructions
1. Clone the Repository:
>git clone [repository-link]
>cd [repository-directory] 

2. Install Required Packages:
>pip install -r requirements.txt 

3. Configure Environment Variables:
Create a .env file with the following content: 
> OPENAI_API_KEY=[your-openai-api-key]


# Running the Application
To launch the Streamlit app, use the following command:
> streamlit run app.py

Below is the interface of the app
Initial interface when the chatbot has not yet received the web link
![example](/Images/one.jpg)

Interface when the chatbot has received the website link
![example](/Images/two.jpg)

conversation interface between user and chatbot
![example](/Images/three.jpg)

If you find the project good, please give me a star, thank you very much.

Happy Coding! 🚀🤖

