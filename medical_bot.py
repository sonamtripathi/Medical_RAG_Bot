import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline

# Loading environment variables
load_dotenv(find_dotenv())

# Vector store path
DB_FAISS_PATH = "vectorstore_db_faiss"

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512"
        }
    )
    return llm

def load_biomedical_llm(HF_TOKEN):
    llm = pipeline(
        "text-generation",
        model="allenai/biomed_roberta_base",
        token=HF_TOKEN
    )
    return llm

def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="💬", layout="wide")

    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>Medical Chatbot App 🏥💬</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align: center;'>Welcome to the Medical Chatbot! Ask any medical-related question and get answers based on a large knowledge base.</p>
        <p><strong>How it works:</strong></p>
        <ul>
            <li><strong>Ask a question:</strong> Type your question in the input box below.</li>
            <li><strong>Context-based answers:</strong> The chatbot first checks the knowledge base for relevant information.</li>
            <li><strong>Fallback to Biomedical Model:</strong> If no relevant information is found, the assistant uses a biomedical model to provide an answer.</li>
        </ul>
        <p>Feel free to ask any medical questions you may have!</p>
        """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>Ask the Medical Chatbot 💡</h1>", unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.chat_message(message['role']).markdown(f"**You:** {message['content']}")
        else:
            st.chat_message(message['role']).markdown(f"**Assistant:** {message['content']}")

    prompt = st.chat_input("Ask a medical question...", key="user_input")

    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message('user').markdown(f"**You:** {prompt}")

        context = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages if msg['role'] != 'user']
        )

        CUSTOM_PROMPT_TEMPLATE = '''
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        '''

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': prompt_template}
            )

            with st.spinner("Thinking... 🤔"):
                response = qa_chain.invoke(prompt)

            result = response["result"]
            source_documents = response["source_documents"]

            used_biomedical_model = False

            if not result or "don't know" in result.lower():
                st.chat_message('assistant').markdown(
                    "**Assistant:** I couldn't find an answer in the knowledge base. Switching to a Biomedical model... 🧬"
                )

                biomedical_llm = load_biomedical_llm(HF_TOKEN)

                with st.spinner("Fetching answer from Biomedical model... 🔬"):
                    biomedical_response = biomedical_llm(
                        f"Context: {context}\nAnswer the following medical question: {prompt}",
                        max_length=256,
                        do_sample=False
                    )

                result = biomedical_response[0]["generated_text"]
                used_biomedical_model = True

                if "don't know" in result.lower() or "unknown" in result.lower():
                    result = "I'm sorry, I don't know the answer to that question."

            st.chat_message('assistant').markdown(f"**Assistant:** {result}")
            st.session_state.messages.append({'role': 'assistant', 'content': result})

            # Only show sources if:
            # 1. Not using biomedical model
            # 2. The result is not "I don't know"
            if not used_biomedical_model and "don't know" not in result.lower() and "i'm sorry" not in result.lower():
                source_docs_text = "\n\n".join([
                    f"**Source {i+1}:** {doc.metadata.get('source', 'No source info available')}\n{doc.page_content}"
                    for i, doc in enumerate(source_documents)
                ])
                if source_docs_text.strip():
                    with st.expander("Source Documents 📚"):
                        st.markdown(source_docs_text)
            elif used_biomedical_model and "i'm sorry" not in result.lower():
                st.info("ℹ️ Note: This answer was generated by a Biomedical model, not retrieved from the knowledge base.")

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
