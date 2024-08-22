
import streamlit as st
import os
import time
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from atlassian import Jira
from langchain.agents import tool
from langchain_core.tools import StructuredTool,BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough

st.html(
    '''
        <style>
            div[aria-label="dialog"]>button[aria-label="Close"] {
                display: none;
            }
        </style>
    '''
)

@st.experimental_dialog("Configuração de Login")
def config():
    st.write(f"Entre com as suas credenciais do Jira:")
    url = st.text_input("Jira URL")
    email = st.text_input("Email")
    api_key = st.text_input("API Key")

    col11, col22 = st.columns([0.5,0.5])
    with col11:
        if st.button("Submit",use_container_width=True):
            st.session_state.credenciais = {"url": url, "email": email,"api_key": api_key}
            st.rerun()
    with col22:
        if st.button("Close",use_container_width=True):
            st.rerun()


def config_jira():
    JIRA_INSTANCE_URL = st.session_state.credenciais["url"]
    USER_NAME = st.session_state.credenciais["email"]
    PASSWORD = st.session_state.credenciais["api_key"]
    print(JIRA_INSTANCE_URL,USER_NAME,PASSWORD)
    jira = Jira(
    url= JIRA_INSTANCE_URL,
    username= USER_NAME,
    password= PASSWORD,
    cloud=True)
    return jira

#if "credenciais" in st.session_state:
#    jira = config_jira()


class IssueJira(BaseModel):
    project: str = Field(description="nome do projeto onde será criado o issue")
    title: str = Field(description="Nome do issue, em frase resumida com até 4 palavras")
    description: str = Field(description="descrição do issue.")
    issuetype: Literal['Task', 'Story', 'Epic'] =  Field(description="Tipo do issue a ser criado")
    #priority: Literal['Highest', 'High', 'Medium','Low','Lowest'] = Field(description="An issue's priority indicates its relative importance. ")

@tool(args_schema=IssueJira)
def criar_issue_Jira(**dict_info:IssueJira) -> dict:
    """Chame essa função para criar issues no Jira"""
    fields = {
        "project":
        {
            "key": dict_info["project"]
        },
        "summary": dict_info["title"],
        "description": dict_info["description"],
        "issuetype": {
            "name": dict_info["issuetype"]
        }
        }
    
    try:    # Create issue
        jira = config_jira()
        result = jira.issue_create(fields)
        print(result)
    except Exception as e:
        print(e)
        if type(e).__name__ in ['MissingSchema','NameError']:
            return "Aconteceu um erro ao enviar o issue ao Jira. O usuário não configurou as credenciais da sua conta do Jira"
        else:
            return f"Aconteceu o erro {type(e).__name__} ao enviar o issue ao Jira"
    
    return result

tools = [criar_issue_Jira]

prompt_2 = ChatPromptTemplate.from_messages([
    ("system", """Você é um especialista em documentação funcional e em Jira.
    Você escreve todo tipo de documentação funcional com base na orientação do usuário.
    Você ajuda ao usuário, em uma jornada, para criar doumentações funcionais desde do nível macro ao micro.
    Quando a documentação estiver a nível de storys, você pode enviálas ao jira, se o usuário solicitar.
    Caso seja solicitado essas informações devem ser enviadas ao Jira.
    Você não pode inventar o nome do projeto do Jira.
    Caso o usuário não informe o nome do projeto, você deve avisá-lo que ele deve adicionar o nome do projeto do jira, para que você possa criar o issue.
    Ao criar um issue você deve me avisar qual é o id e a Key do issue, assim como o link para que eu possa abrir-lo no Jira
    Para casos de escrita de de histórias/Storys, você deve utilizar o formato BDD, com cenários e critérios de aceite.
    As storys precisam ter formatação markdown na sua descrição.
     """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

OPENAI_API_KEY = os.environ['OPEN_API_KEY']

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY,temperature=0.5)

model_jira_with_tool_test = model.bind(functions=[format_tool_to_openai_function(criar_issue_Jira)])

agent_chain_2 = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt_2 | model_jira_with_tool_test | OpenAIFunctionsAgentOutputParser()

@st.cache_resource()
def memory():
    memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
    return memory

memory = memory()
agent_executor = AgentExecutor(agent=agent_chain_2, tools=[criar_issue_Jira], verbose=True, memory=memory)


# Streamed response emulator
def response_generator(response):
    time.sleep(0.05)
    for word in response.split(" "):
        yield word+ " "
        time.sleep(0.05)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.container():
    col1, col2 = st.columns([0.8,0.2])
    with col1:
        st.title("Agente Funcional")
    with col2:
        if st.button(":gear:",use_container_width=True):
            config()


initial_message = st.chat_message("assistant")
initial_message.write(response_generator("Olá, como posso ajudá-lo hoje?"))  

for message in st.session_state.messages:
    if message['role'] == "assistant":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    

if prompt := st.chat_input("Faça uma pergunta?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(''):
            response=agent_executor.invoke({'input':prompt})["output"]
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
