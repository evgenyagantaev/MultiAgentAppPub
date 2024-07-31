import getpass
import os
import time

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START

from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_openai import ChatOpenAI
import functools
from langchain_core.messages import AIMessage

from openai import InternalServerError

import asyncio
from aioconsole import ainput
import msvcrt

import json
from datetime import datetime


os.environ["OPENAI_API_KEY"] = "ключ"
os.environ["TAVILY_API_KEY"] = "ключ"
os.environ["OPENAI_API_BASE"] = "адрес"

llm1 = ChatOpenAI(model="openai/gpt-4o-mini", max_tokens=13000)
llm2 = ChatOpenAI(model="openai/gpt-4o-mini", max_tokens=13000)
llm3 = ChatOpenAI(model="openai/gpt-4o", max_tokens=13000)
llm4 = ChatOpenAI(model="anthropic/claude-3.5-sonnet", max_tokens=13000)
# llm3 = ChatOpenAI(model="deepseek/deepseek-coder", max_tokens=13000)
# llm4 = ChatOpenAI(model="deepseek/deepseek-coder", max_tokens=13000)
llm5 = ChatOpenAI(model="nvidia/nemotron-4-340b-instruct", max_tokens=13000)
llm6 = ChatOpenAI(model="meta-llama/llama-3.1-405b-instruct", max_tokens=13000)

def get_chat_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chat_history_{timestamp}.json"

def append_to_chat_history(message):
    filename = getattr(append_to_chat_history, 'filename', None)
    if filename is None:
        filename = get_chat_filename()
        append_to_chat_history.filename = filename
        # Создаем файл и инициализируем его пустым списком
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([], f)
    
    with open(filename, "r+", encoding="utf-8") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = []
        
        history.append(message)
        
        f.seek(0)
        json.dump(history, f, ensure_ascii=False, indent=2)
        f.truncate()

def check_escape():
    for _ in range(3):  # Цикл 3 повтора
        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key == b'\x1b'  # Код клавиши Escape
        time.sleep(0.1)
    return False  # Интервал 0.1 секунды

def human_input():
    return input("Мастер: ")

@tool
def human_input_tool():
    """Использовать для получения ввода от человека."""
    return human_input()

MAX_MESSAGES = 9  
def human_node(state, name):
    # Ограничиваем количество сообщений
    state['messages'] = state['messages'][-MAX_MESSAGES:]
    
    # Удаляем все сообщения с нулевым контентом
    state['messages'] = [msg for msg in state['messages'] if msg.content.strip()]
    
    # Заменяем все ToolMessage на ChatMessage с ролью 'user'
    state['messages'] = [
        ChatMessage(content=msg.content, role='user') if isinstance(msg, ToolMessage) else msg
        for msg in state['messages']
    ]
    
    human_message = human_input()
    return {
        "messages": [HumanMessage(content=human_message)],
        "sender": "master"
    }

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant, collaborating with other assistants."
            "Предпочтительный язык взаимного общения у нас в команде - русский"
            "Список команды:" 
            "Евгений (системный ник master) модератор-человек;"
            "Супер (системный ник supervisor) LLM Anthropic 3.5 Sonet"
            "Умник (системный ник reviewer) LLM OpenAi GPT 4o"
            " Use the provided tools to progress towards answering the question."
            " If you are unable to fully answer, that's OK, another assistant with different tools "
            " will help where you left off. Execute what you can to make progress."
            " If you or any of the other assistants have the final answer or deliverable,"
            " prefix your response with FINAL ANSWER so the team knows to stop."
            " You have access to the following tools: {tool_names}.\n{system_message}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    agent = prompt | llm.bind_tools(tools)
    return agent, prompt

def create_supervisor_agent(llm, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant, collaborating with other assistants."
            "Предпочтительный язык взаимного общения у нас в команде - русский"
            "Список команды:" 
            "Евгений (системный ник master) модератор-человек;"
            "Супер (системный ник supervisor) LLM Anthropic 3.5 Sonet"
            "Умник (системный ник reviewer) LLM OpenAi GPT 4o"
            " If you or any of the other assistants have the final answer or deliverable,"
            " prefix your response with FINAL ANSWER so the team knows to stop."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
    )
    prompt = prompt.partial(system_message=system_message)
    agent = prompt | llm
    return agent, prompt

tavily_tool = TavilySearchResults(max_results=5)

# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()
@tool
def python_repl(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER.")

# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Helper function to create a node for a given agent
from langchain_core.messages import AIMessage

def agent_node(state, agent, prompt, name):
    # Ограничиваем количество сообщений
    state['messages'] = state['messages'][-MAX_MESSAGES:]
    
    # Удаляем все сообщения с нулевым контентом
    state['messages'] = [msg for msg in state['messages'] if msg.content.strip()]
    
    # Заменяем все ToolMessage на ChatMessage с ролью 'user'
    state['messages'] = [
        ChatMessage(content=msg.content, role='user') if isinstance(msg, ToolMessage) else msg
        for msg in state['messages']
    ]
    
    # print(f"\nDebug: Input state for {name}:")
    # print(f"Messages: {state['messages']}")
    # print(f"Sender: {state.get('sender', 'Not specified')}")
    # print(f"\nDebug: full prompt for {name}:")
    # print(prompt)
    
    try:
        result = agent.invoke(state)
    except InternalServerError as e:
        print(f"Ошибка сервера при вызове агента {name}: {str(e)}")
        return {
            "messages": [AIMessage(content=f"Ошибка: Не удалось получить ответ от {name}. Пожалуйста, попробуйте еще раз позже.")],
            "sender": name,
        }
    
    # print(f"\nDebug: Output from {name}:")
    # print(f"Result: {result}")
    
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name", "response_metadata", "id", "usage_metadata"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

master_node = functools.partial(human_node, name="master")

junior, junior_prompt = create_agent(
    llm1,
    [],
    system_message="Твоё имя Джун. Системный ник junior. Предпочтительный язык - русский."
                   "Если ты не можешь добавить ничего нового, важного, интересного и содержательного,"
                   "ты должен начать очередное сообщение с кодовой фразы FINAL ANSWER",
)
junior_node = functools.partial(agent_node, agent=junior, prompt=junior_prompt, name="junior")

intern, intern_prompt = create_agent(
    llm2,
    [],
    system_message="Твоё имя Интерн. Системный ник intern. Предпочтительный язык - русский."
                   "Если ты не можешь добавить ничего нового, важного, интересного и содержательного,"
                   "ты должен начать очередное сообщение с кодовой фразы FINAL ANSWER",
)
intern_node = functools.partial(agent_node, agent=intern, prompt=intern_prompt, name="intern")

reviewer, reviewer_prompt = create_agent(
    llm3,
    [],
    system_message="Твоё имя Умник. Системный ник reviewer. Предпочтительный язык - русский." 
                   "Твоя задача - очень критично рассматривать предложения коллег, находить в них слабые места,"
                   "указывать на них и предлагать альтернативные решения, исправляющие указанные недостатки."
                   "Если ты не можешь добавить ничего нового, важного, интересного и содержательного,"
                   "ты должен начать очередное сообщение с кодовой фразы FINAL ANSWER",
)
reviewer_node = functools.partial(agent_node, agent=reviewer, prompt=reviewer_prompt, name="reviewer")

supervisor, supervisor_prompt = create_supervisor_agent(
    llm4,
    system_message="Твоё имя Супер. Системный ник supervisor. Предпочтительный язык - русский." 
                   "Твоя функция - всесторонне анализировать поставленную задачу, предлагать пути её решения,"
                   "рассматривать предложения коллег, принимать хорошие предложения и аргументированно отвергать неудачные;"
                   "За тобой последнее слово, ты формулируешь окончательное решение. Как только ты принял окончательное решение,"
                   "или тебе требуется вмешательство человека в обсуждение, ты должен начать очередное сообщение с кодовой фразы FINAL ANSWER"
                   "Если ты получил от коллеги сообщение с кодовой фразой FINAL ANSWER и тебе самому нечего добавить," 
                   "ты должен завершить обсуждение, начав очередное сообщение с кодовой фразы FINAL ANSWER"
)
supervisor_node = functools.partial(agent_node, agent=supervisor, prompt=supervisor_prompt, name="supervisor")

from langgraph.prebuilt import ToolNode

tools = [tavily_tool, python_repl]
tool_node = ToolNode(tools)

# Either agent can decide to end
from typing import Literal

# def router(state) -> Literal["call_tool", "__end__", "continue"]:
#     # This is the router
#     messages = state["messages"]
#     last_message = messages[-1]
#     # print(f"\nDebug: last message on router:")
#     # print(last_message)
#     if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls'):
#         if last_message.tool_calls:
#             # The previous agent is invoking a tool
#             return "call_tool"
#     elif isinstance(last_message, HumanMessage):
#         # Обработка сообщения от человека
#         "continue"
#     elif "FINAL ANSWER" in last_message.content:
#         # Any agent decided the work is done
#         return "master"
#     return "continue"

#def router(state) -> Literal["call_tool", "continue", "master"]:
def router(state) -> Literal["continue", "master"]:
    if check_escape():
        return "master"
    
    messages = state["messages"]
    last_message = messages[-1]
    
    if "FINAL ANSWER" in last_message.content:
        return "master"  # После завершения передаем управление выше по иерархии
    return "continue"

workflow = StateGraph(AgentState)

workflow.add_node("master", master_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("reviewer", reviewer_node)
#workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "supervisor",
    router,
    {"continue": "reviewer", "master": "master"},
)
workflow.add_conditional_edges(
    "reviewer",
    router,
    {"continue": "supervisor", "master": "supervisor"},
)

# workflow.add_conditional_edges(
#     "call_tool",
#     # Each agent node updates the 'sender' field
#     # the tool calling node does not, meaning
#     # this edge will route back to the original agent
#     # who invoked the tool
#     lambda x: x["sender"],
#     {
#         "junior": "junior",
#         "intern": "intern",
#     },
# )

workflow.add_edge("master", "supervisor")

workflow.set_entry_point("master")

#workflow.add_edge(START, "junior")

graph = workflow.compile()

from IPython.display import Image
from PIL import Image
import io

try:
    # Получаем PNG-данные
    png_data = graph.get_graph(xray=True).draw_mermaid_png()
    
    # Создаем объект изображения Pillow из данных
    img = Image.open(io.BytesIO(png_data))
    
    # Сохраняем изображение
    img.save('graph.png')
    #print("Схема графа сохранена как 'graph.png'")
except Exception as e:
    print(f"Не удалось сохранить изображение. Ошибка: {e}")
    
try:
    img_data = graph.get_graph(xray=True).draw_mermaid_png()
    img = Image.open(io.BytesIO(img_data))
    img.show()
except Exception as e:
    print(f"Не удалось отобразить изображение. Ошибка: {e}")


# # for s in events:
# #     print(s)
# #     print("----")
# #     time.sleep(2)

events = []
print("Начинаем работу над задачей.")
print()
events = graph.stream({"messages": [], "sender": "master"}, {"recursion_limit": 999999},)

for s in events:
    for event_data in s.items():
        messages = event_data[1]['messages']
        sender = event_data[1].get('sender', "")
        
        for message in messages:
            if isinstance(message, HumanMessage):
                content = message.content
                append_to_chat_history({"role": "human", "content": content})
            elif isinstance(message, AIMessage):
                content = message.content
                print(f"{sender}: {content}")
                append_to_chat_history({"role": sender, "content": content})
            else:
                content = str(message)
                print(f"{sender}: {content}")
                append_to_chat_history({"role": sender, "content": content})
    
    print()
    print("----")
    print()
    time.sleep(2)