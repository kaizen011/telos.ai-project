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