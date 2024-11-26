import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from pathlib import Path
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


###########################################################################################

SYS_PROMPT = '''
You are a helpful assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer or the context is unrelevant to the question, just say that you don't know. \
Keep your answer concise. \

Context: \
{context}
'''

LLM_MODEL = 'adrienbrault/nous-hermes2pro-llama3-8b:q8_0'

EMBED_MODEL = 'Alibaba-NLP/gte-base-en-v1.5'

INPUT_DATA_DIR = Path(f"./data")


###########################################################################################


# Fix for some new weird "no attribute 'verbose'" bug 
# https://github.com/hwchase17/langchain/issues/4164
langchain.verbose = False


@st.cache_resource
def load_chunks():
    """
    Load and split chunks of text from PDF files in the input directory.
    """
    loader = PyPDFDirectoryLoader(INPUT_DATA_DIR)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)
    return chunks


@st.cache_resource
def create_knowledge_base():
    """
    Create a knowledge base using vector embeddings from loaded chunks and return an in-memory Qdrant instance.
    """
    chunks = load_chunks()

    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL,
                                    model_kwargs={'trust_remote_code': True})

    knowledge_base = Qdrant.from_documents(
        chunks,
        embeddings,
        location=":memory:",
        collection_name="doc_chunks",
    )

    return knowledge_base


# Set up LLM and QA Chain
llm = Ollama(
    model=LLM_MODEL,
    temperature=0.1,
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
)
chain = create_stuff_documents_chain(llm, qa_prompt)


# Page setup
st.set_page_config(page_title="LLM-RAG")
st.header("RAG-LLM Chatbot 💬")

# Process docs
knowledge_base = create_knowledge_base()


# Conversation 
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_conversation():
    st.session_state.messages = []

if user_input := st.chat_input("Ask a question:"):
    # Display conversation history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    docs = knowledge_base.similarity_search(user_input, k=5, score_threshold=0.5)

    references = [f"{doc.metadata['source'].split(str(INPUT_DATA_DIR)+'/')[1]} - p.{doc.metadata['page']}" for doc in docs]
    references = "  \n  \n**References:**  \n" + "  \n".join([f"[{i}]({references[i]})" for i in range(len(references))])

    st.chat_message("user").markdown(user_input)

    # Grab and print response
    with st.spinner('Please wait...'):

        response = chain.invoke({
            "context": docs, 
            "question": user_input, 
            "history": [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
        }) 
        

        output_text = response + references

        st.chat_message("assistant").markdown(output_text)
        

        st.session_state.messages.append({"role": "user", "content": user_input })
        st.session_state.messages.append({"role": "assistant", "content": output_text})


    st.button("Clear Conversation", on_click=reset_conversation)
        

