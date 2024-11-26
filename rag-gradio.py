
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path
import gradio as gr
import csv


###########################################################################################
SYS_PROMPT = '''
You are a helpful assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
Dont leave any relevant information out while answering, use all relevant context. \
If the context is missing or you don't know the answer, just say that information doesn't exist in your knowledge base. \
Keep your answer concise, detailed and informative. \

'''


USER_PROMPT = '''
Context: \
{context}

Question: \
{question}

Answer: \
'''

LLM_MODEL = 'llama3.1:8b'

EMBED_MODEL = 'Alibaba-NLP/gte-base-en-v1.5'

INPUT_DATA_DIR = Path(f"./data")


###########################################################################################


# Fix for some new weird "no attribute 'verbose'" bug 
# https://github.com/hwchase17/langchain/issues/4164
langchain.verbose = False



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

###############################################################################################


# Set up LLM and QA Chain
llm = Ollama(
    model=LLM_MODEL,
    temperature=0.1,
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        MessagesPlaceholder("history", optional=True),
        ("human", USER_PROMPT),
    ]
)
chain = create_stuff_documents_chain(llm, qa_prompt)

knowledge_base = create_knowledge_base()

###############################################################################################

def process_input(user_input, history=[]):
    """
    Process user input and return a response.
    """
    # Save the user input to a CSV file
    with open('questions.csv', 'a', newline='') as csvfile:
        fieldnames = ['question']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({'question': user_input})
    

    msg_history = []
    for i in range(len(history)):
        if i % 2 == 1:
            msg_history.append({'role': 'human', 'content': history[i][0]})
        else:
            msg_history.append({'role': 'ai', 'content': history[i][1].split('**References:**')[0]})

    

    docs = knowledge_base.similarity_search(user_input, k=10, score_threshold=0.6)
    
    if not docs:
        # Get the last 2 questions from the history
        last_2_questions = [msg['content'] for msg in msg_history if msg['role'] == 'human'][-2:]
        search_query =''.join([user_input] + last_2_questions)

        docs = knowledge_base.similarity_search(search_query, k=10, score_threshold=0.5)

        if not docs:
            return "This information doesn't exist in my knowledge base."

    references = [f"{doc.metadata['source'].split(str(INPUT_DATA_DIR)+'/')[1]}, p.{doc.metadata['page']}" for doc in docs]
    references = "  \n  \n**References:**  \n" + "  \n".join([f"[{i+1}] {references[i]}" for i in range(len(references))])

    response = chain.invoke({
        "context": docs, 
        "question": user_input, 
        "history": msg_history
    }) 

    response += references


###############################################################################################

iface = gr.ChatInterface(
    chatbot=gr.Chatbot(
        layout='bubble',
        height="64vh",
    ),
    fn=process_input,
    title="RAG - LLM Chatbot 💬",
    theme="soft",
    retry_btn=None,
    undo_btn=None,
    fill_height=True, 
)

if __name__ == "__main__":
    iface.launch(share=True)