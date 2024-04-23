import asyncio

from dotenv import load_dotenv

import PyPDF2

from langchain_community.embeddings import OllamaEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Qdrant

from langchain.chains import ConversationalRetrievalChain

from langchain_community.chat_models import ChatOllama

from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl

import os

import qdrant_client

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

load_dotenv()

@cl.on_chat_start

async def on_chat_start():

    load_dotenv()

 

    loader = PyPDFLoader("mental_health_Document.pdf")

    documents=loader.load()

    # Split the text into chunks

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    texts = text_splitter.split_documents(documents)

 

    model_name = "BAAI/bge-large-en"

    model_kwargs = {'device': 'cpu'}

    encode_kwargs = {'normalize_embeddings': False}

 

    embeddings = HuggingFaceBgeEmbeddings(

    model_name = model_name,

    model_kwargs = model_kwargs,

    encode_kwargs = encode_kwargs

   )  

    

    qdrant_host =  os.environ['QDRANTHOST']
    qdrant_apikey =  os.environ['QDRANTAPIKEY']
    qdrant_collection_name = os.environ['QDRANT_COLLECTION_NAME']
  
  

    qdrant = Qdrant.from_documents(

    texts,

    embeddings,

    url =  qdrant_host,

    prefer_grpc = False,

    collection_name = qdrant_collection_name,

    api_key= qdrant_apikey

    )

 

    # Initialize message history for conversation

    message_history = ChatMessageHistory()

 

    # Memory for conversational context

    memory = ConversationBufferMemory(

        memory_key="chat_history",

        output_key="answer",

        chat_memory=message_history,

        return_messages=True,

    )

    # Create a chain that uses the Qdrant vector store

    chain = ConversationalRetrievalChain.from_llm(

        ChatOllama(model="mistral"),

        # chain_type="stuff",

        retriever=qdrant.as_retriever(),

        memory=memory,

        return_source_documents=True,

    )

 

    # Let the user know that the system is ready

    await cl.Message(content=f"Processing  done. You can now ask questions!").send()

 

    # Store the chain in user session

    cl.user_session.set("chain", chain)

 

 

@cl.on_message

async def main(message: cl.Message):

    # Initialize variables outside the try block

    source_documents = None  # Initialized to None or an empty list

    

    print("Retrieving chain from session...")

    chain = cl.user_session.get("chain")

    

    print("Creating async callback handler...")

    cb = cl.AsyncLangchainCallbackHandler()

 

    try:

     print("Invoking chain with user's message content...")

     res = await chain.ainvoke(message.content, callbacks=[cb])

    # Immediately inspect the raw response

     print("Raw search response:", res)

 

     answer = res.get("answer", "Sorry, I couldn't process your request.")

     source_documents = res.get("source_documents", [])

    

    # Debug print to confirm the structure and content

     print('Source documents received:', source_documents)

    except Exception as e:

     print("Error during search or document processing:", e)

     answer = "An error occurred: " + str(e)

 

    text_elements = []  # Initialize list to store text elements

 

    # Process source documents if available

    if source_documents:

     for source_idx, source_doc in enumerate(source_documents):

        # Assuming source_doc is an instance of a Document class with a page_content property

        if hasattr(source_doc, 'page_content'):  # Check if the page_content property exists

            source_name = f"source_{source_idx}"

            text_elements.append(

                cl.Text(content=source_doc.page_content, name=source_name)  # Access the property directly

            )

            

    else:

        print("Warning: No source documents found.")

    

    # Return results

    await cl.Message(content=answer, elements=text_elements).send()

 

 

 

if __name__ == '__main__':

    asyncio.run(on_chat_start())