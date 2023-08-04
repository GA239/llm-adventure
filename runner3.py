from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from runner2 import room_chain

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(raise_error_if_not_found=True))

llm = ChatOpenAI(temperature=0)
llm1 = OpenAI(temperature=0)
llm_math_chain = LLMMathChain(llm=llm1, verbose=True)

riddle = {'riddle': "What is the programming language whose syntax doesn't need semicolons?", 'answer': 'Python'}
conversation = room_chain(topic="programming languages, data structures, and algorithms", riddle=riddle)

tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    # Tool(
    #     name="Assistant",
    #     func=conversation.run,
    #     description="Use if don't know, what to use",
    # )
]

prefix = """
Have a conversation with a human, answering the following questions as best you can.
If you can't answer a question using tools, just say "I don't know". 
You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"]
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

agent_chain.run(input="How many people live in canada?")
agent_chain.run(input="Hi, I'm a human. What is your name?")
# agent_chain.run(input="Calculate 2+2")
