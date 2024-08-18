from uuid import UUID
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import os
from langchain import PromptTemplate,LLMChain
from langchain.chains import ConversationChain
import requests
from langchain_community.utilities import SerpAPIWrapper
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import pickle
!pip install pickle


import getpass
import os


st.set_page_config(page_title="Travel Guru")

with st.sidebar:
    HugginngFaceAPI = st.text_input("HuggingFaceHub Key", type="password")
    Search_api= st.text_input("SerpAPI key(Required for web search)",type="password")
    
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            huggingfacehub_api_token=HugginngFaceAPI

)

with open("https://github.com/shadow492/LLM_Chatbot_Assistant/blob/main/Book_data.pickle", "rb") as f:
    db = pickle.load(f)

memory = ConversationBufferMemory(memory_key='chat_history',
                                  return_messages=True,
                                  output_key="output")


from langchain.agents import AgentType,initialize_agent
from langchain.tools import StructuredTool, Tool,tool,BaseTool
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage


st.header("Hello there, welcome to Book Guru, your personal book guide")

class StreamHandler(BaseCallbackHandler):
    def __init__(self,container,initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str,**kwargs) -> None:
        self.text +=token
        self.container.markdown(self.text)


search = SerpAPIWrapper(serpapi_api_key=Search_api)
tools = [
    Tool.from_function(
        func = search.run,
        name = "Search",
        description = "Useful for when you need to answer questions about current events"),
        create_retriever_tool(
            db.as_retriever(),
            "Philosopher",
            "searches and returns the most authentic answers from the database"
        )    
]


from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
agent_executor = create_conversational_retrieval_agent(llm,tools,memory_key="chat_history",verbose = True)

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role= "assistant", content= "How can I help you?")]


if "memory" not in st.session_state:
    st.session_state['memory'] = memory

for message in st.session_state["messages"]:
    st.chat_message(message.role).write(message.content)


if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not HugginngFaceAPI:
        st.info("Please add your HuggingFaceHub API key to continue.")
        st.stop()
    


    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        Stream_handler = StreamHandler(st.empty())
        response = agent_executor.invoke({"input":st.session_state.messages})
        if isinstance(response, dict):
            content = response.get('output', str(response))
        else:
            content = str(response)

        st.session_state.messages.append(ChatMessage(role="assistant", content= response.get("output")))
        st.write(response.get("output"))



