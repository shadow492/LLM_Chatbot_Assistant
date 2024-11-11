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
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.utilities import SerpAPIWrapper
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain import hub

import getpass
import os


st.set_page_config(page_title="Travel Guru")
st.header("Hello there, welcome to Book Guru, your personal book guide")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system = '''Respond to the human as helpfully,accurately and efficiently as possible. You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format and donot iterate too much for a query:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'''

human = '''{input}

{agent_scratchpad}

(reminder to respond in a JSON blob no matter what)
'''

human = '''{input}

{agent_scratchpad}

(reminder to respond in a JSON blob no matter what)'''

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
    ]
)

left, middle = st.columns(2, vertical_alignment="center")
with left.popover("HuggingFace",use_container_width=True):
    HugginngFaceAPI = st.text_input("HuggingFaceHub Key", type="password",)
    
with middle.popover("SerpAPI",use_container_width=True):
    Search_api= st.text_input("SerpAPI key(Required for web search)",type="password")

if not HugginngFaceAPI :
    st.info("Please add your API keys to continue.", icon="🗝️")


if not Search_api :
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    llm = HuggingFaceEndpoint(
            repo_id="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            huggingfacehub_api_token=HugginngFaceAPI

    )

    memory = ConversationBufferMemory(memory_key='chat_history',
                                  return_messages=False,
                                  output_key="output")


    from langchain.agents import AgentType,initialize_agent
    from langchain.tools import StructuredTool, Tool,tool,BaseTool
    from langchain.agents.agent_toolkits import create_retriever_tool
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import ChatMessage
    from langchain.prompts import load_prompt

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    class StreamHandler(BaseCallbackHandler):
        def __init__(self,container,initial_text=""):
            self.container = container
            self.text = initial_text
        def on_llm_new_token(self, token: str,**kwargs) -> None:
            self.text +=token
            self.container.markdown(self.text)
    python_repl = PythonREPL()
    tools = [
      ]

    

    agent = create_structured_chat_agent(llm, tools,prompt=prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools,memory_key="chat_history",verbose=True,return_only_output=True,handle_parsing_errors=True,early_stopping_method='generate')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role= "assistant", content= "How can I help you?")]

    if "memory" not in st.session_state:
        st.session_state['memory'] = memory

    for message in st.session_state["messages"]:
        st.chat_message(message.role).write(message.content)


    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            Stream_handler = StreamHandler(st.empty())
            response = agent_executor.invoke({"input":st.session_state.messages},return_only_outputs=True,)
            if isinstance(response, dict):
                content = response.get('output', str(response))
            else:
                content = str(response)

            st.session_state.messages.append(ChatMessage(role="assistant", content= response.get("output")))
            st.write(response.get("output"))


else: 
    
    
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            huggingfacehub_api_token=HugginngFaceAPI

    )

    memory = ConversationBufferMemory(memory_key='chat_history',
                                  return_messages=False,
                                  output_key="output")


    from langchain.agents import AgentType,initialize_agent
    from langchain.tools import StructuredTool, Tool,tool,BaseTool
    from langchain.agents.agent_toolkits import create_retriever_tool
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import ChatMessage




    class StreamHandler(BaseCallbackHandler):
        def __init__(self,container,initial_text=""):
            self.container = container
            self.text = initial_text
        def on_llm_new_token(self, token: str,**kwargs) -> None:
            self.text +=token
            self.container.markdown(self.text)

    search = SerpAPIWrapper(serpapi_api_key=Search_api)
    python_repl = PythonREPL()
    tools = [
        Tool.from_function(
            func = search.run,
            name = "Search",
            description = "Useful for when you need to answer questions about current events and unknown information"),
        
    ]

    agent = create_structured_chat_agent(llm, tools,prompt=prompt)
    agent_executor = AgentExecutor(agent=agent,tools=tools,memory_key="chat_history",verbose=True,return_only_output=True,handle_parsing_errors=True,early_stopping_method='generate')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role= "assistant", content= "How can I help you?")]

    if "memory" not in st.session_state:
        st.session_state['memory'] = memory

    for message in st.session_state["messages"]:
        st.chat_message(message.role).write(message.content)


    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            Stream_handler = StreamHandler(st.empty())
            response = agent_executor.invoke({"input":st.session_state.messages},return_only_outputs=True,)
            if isinstance(response, dict):
                content = response.get('output', str(response))
            else:
                content = str(response)

            st.session_state.messages.append(ChatMessage(role="assistant", content= response.get("output")))
            st.write(response)
