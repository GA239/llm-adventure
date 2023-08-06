from __future__ import annotations

import json
from typing import Dict, Any

import langchain
from dotenv import load_dotenv, find_dotenv
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.cache import InMemoryCache
from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
from langchain.memory import ConversationSummaryMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field

from adventure.utils import get_model

load_dotenv(find_dotenv(raise_error_if_not_found=True))
__DEBUG__ = True

ADVENTURE_GAME_ROOM_PROMPT_TEMPLATE = """
You are an expert in {topic}. You are following the following rules:
- You must not refer to yourself in any way.
- You can only answer questions that are asked in the context of the game.
- You can't pretend a player.
- Be short and precise in your answers.

You also follow the rules of the Adventure Game Room.
The rules of Adventure Game Room:

- You are talking with a player. 
- The player is trying to guess a riddle.
- You have a riddle that player need to find the solution to. The riddle: "{riddle}". 
- The riddle is related to {topic}.
- You have a correct answer to the riddle. The correct answer is "{answer}".
- The player can ask you questions about the riddle. You can answer the questions about riddle, but don't say the answer.
- The player can ask you questions about the answer. Don't provide answer. You can only give clues or hints
- You NEVER say the precise answer to the player. You can only give clues
- You should choose action from available actions based on the players' input and history of conversation.

The following actions are available.
{actions}
- You should choose the next state from available states based on the action.

The following states are available
{states}
"""

states_str = f"""
* guessing_riddle: if player makes a guess, compare it with the correct answer. If player, asked a question, answer it by giving a clue.
* finished: you should stop the game
"""

actions_str = """
* start_game: only in the initial state, start the game by giving player the riddle
* guess_correct: if player guessed correctly. The next state is "finished"
* guess_incorrect: if player guessed incorrectly. You should give player a clue. The next state is "guessing_riddle"
"""

history_concatenation = """Current conversation:
Player: ```{input}```
"""


class Riddle(BaseModel):
    """A riddle that need to be solved."""
    riddle: str = Field(description="the riddle as a question")
    answer: str = Field(description="the answer to the riddle")


class RoomLoopAnswer(BaseModel):
    """Room Loop Answer"""
    action: str = Field(description="The action to take. Must be one of the valid actions")
    similarity: float = Field(
        description="the similarity of the main idea of the player's answer to the main idea of the correct answer as a value between 0 and 1")
    answer_idea: str = Field(description="the main idea of the player's answer")
    correct_answer_idea: str = Field(description="the main idea of the correct answer")
    reply: str = Field(
        description="Your reply to player. You must say clues in your own words instead of just copying them")
    new_state: str = Field(description="The new state, after this action. Must be one of the valid states")


RoomLoopAnswerParser = PydanticOutputParser(pydantic_object=RoomLoopAnswer)
RiddleParser = PydanticOutputParser(pydantic_object=Riddle)


def room_chain(topic: str = "programming", riddle: dict = None):
    langchain.llm_cache = InMemoryCache()

    model_name = "OpenAI"
    # model_name = "Replicate"
    # model_name = "Cohere"
    # model_name = "HuggingFace_google_flan"
    # model_name = "HuggingFace_mbzai_lamini_flan"
    # model_name = "Local_gpt2"
    # model_name = "Local_lama"
    llm = get_model(model_name, temperature=0.3)

    system_prompt = ADVENTURE_GAME_ROOM_PROMPT_TEMPLATE.format(
        topic=topic,
        riddle=riddle["riddle"],
        answer=riddle["answer"],
        actions=actions_str,
        states=states_str.format(topic=topic)
    )

    template = system_prompt + history_concatenation + "\n{format_instructions}\n"

    prompt = PromptTemplate(
        template=template,
        # input_variables=["history", "input"],
        input_variables=["input"],
        partial_variables={"format_instructions": RoomLoopAnswerParser.get_format_instructions()},
    )
    memory = ConversationSummaryMemory(llm=llm)
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    return conversation


def execute_room_semi_loop(conversation: LLMChain, player_input: str):
    """Parse the answer from the room loop"""
    for _ in range(2):  # couple of attempts to get a valid response
        try:
            reply = conversation.run(player_input)
            rpl_dict = RoomLoopAnswerParser.parse(reply)
            for k, v in rpl_dict.dict().items():
                print(f"Debug::: {k} --> {v}")
            return rpl_dict.dict()
        except json.decoder.JSONDecodeError as e:
            print(e)
            continue


def room_game_loop(topic: str = "programming"):
    riddle = generate_riddle(topic=topic)
    riddle = riddle.dict()
    print(f"Debug::: {riddle}")
    conversation = room_chain(topic=topic, riddle=riddle)

    start = execute_room_semi_loop(conversation, "Hello!")
    print(start["reply"])
    while 1:
        # TODO: add attempts counter
        rpl = execute_room_semi_loop(conversation, input(">>>"))
        reply = rpl["reply"]
        print(reply)

        action, state = rpl["action"], rpl["new_state"]
        if state == "finished":
            break

        similarity = float(rpl["similarity"])
        if similarity > 0.65:
            print("However, Your guess is almost correct!")
            answer = riddle["answer"]
            print(f"The answer is {answer}")
            break

    print("Thank you for playing!, "
          "You can play again in the same room or in another room")


def get_riddle_generator_chat():
    sys_prompt = """
    You are a world class algorithm for generating riddles. 
    Your task is to generate a riddle about {topic}.
    Your knowledge of {topic} should be used to generate riddle.
    
    Hint: The riddle should be short.
    Hint: The riddle should not contain the answer.
    """
    model_name = "ChatOpenAI"
    llm = get_model(model_name, temperature=0.8)

    prompt_msgs = [
        SystemMessagePromptTemplate.from_template(
            sys_prompt, input_variables=["topic"]
        )
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs, input_variables=["topic"], )
    chain = create_structured_output_chain(
        Riddle, llm, prompt,
        verbose=__DEBUG__
    )
    return chain


def get_riddle_generator() -> LLMChain:
    sys_prompt = """
    You are a world class algorithm for generating riddles. 
    Your task is to generate a riddle about {topic}.
    Your knowledge of {topic} should be used to generate riddle.
    
    Hint: The riddle should be short.
    Hint: The riddle should not contain the answer.
    """
    model_name = "OpenAI"
    llm = get_model(model_name, temperature=0.8)

    prompt = PromptTemplate(
        template=sys_prompt + "\n{format_instructions}\n",
        input_variables=["topic"],
        partial_variables={"format_instructions": RiddleParser.get_format_instructions()},
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=__DEBUG__)
    return chain


def generate_riddle(topic="programming", chat_model=False, *args, **kwargs) -> dict[str, Any] | BaseModel | Any:
    chain = get_riddle_generator_chat() if chat_model else get_riddle_generator()
    for _ in range(2):
        try:
            if chat_model:
                return chain.run(topic=topic)
            return RiddleParser.parse(chain.run(topic=topic))
        except json.decoder.JSONDecodeError:
            continue
    raise ValueError("Can't parse the output")


if __name__ == "__main__":
    # r = generate_riddle(topic="programming languages, data structures, and algorithms")
    # print(r)
    # r = generate_riddle(topic="programming languages, data structures, and algorithms", chat_model=True)
    # print(r)
    # room_game_loop(topic="programming languages, data structures, and algorithms")
    room_game_loop(topic="Astronomy")
