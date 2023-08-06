from typing import Any

import langchain
from dotenv import load_dotenv, find_dotenv
from langchain import LLMChain, LLMMathChain, PromptTemplate
from langchain.cache import InMemoryCache
from langchain.chains import SequentialChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from adventure.utils import get_model
from runner2 import execute_room_semi_loop

__DEBUG__ = True
load_dotenv(find_dotenv(raise_error_if_not_found=True))

# - The player is trying to guess a riddle.
# - You have a riddle that player need to find the solution to. The riddle: "{riddle}".
# - The riddle is related to {topic}.
# - You have a correct answer to the riddle. The correct answer is "{answer}".
# - You can answer the questions about riddle, but don't provide the answer.

ADVENTURE_GAME_MATH_ROOM_PROMPT_TEMPLATE = """
You are following the following rules:
- You are playing the question/answer game.
- You play the role an expert in {topic}.
- Your task is to check the correctness of the player's answers.
- You must not refer to yourself in any way. Only to role.
- You are talking with a player.
- You can only answer questions that are asked in the context of the game.
- You can't pretend a player.
- Be short and precise in your answers.

You also follow the rules of the Adventure Game Math Room.
The rules of Adventure Game Math Room:

- The player is trying to solve a math problem.
- You have a math problem that player need to solve. The math problem: "{riddle}".
- You have a correct answer to the math problem. The correct answer is "{answer}". It's a number.

- Don't provide answer. You can only give clues or hints.
- You NEVER say the precise answer to the player. You can only give clues
- You should choose action from available actions based on the players' input and history of conversation.

The following actions are available.
{actions}
- You should choose the next state from available states based on the action.

The following states are available
{states}
"""

states_str = f"""
* guessing_riddle: if player makes a guess, consider player's input to number and compare it with the correct answer. If player, asked a question, answer it by giving a clue.
* finished: you should stop the game
"""

actions_str = """
* start_game: only in the initial state, start the game by giving player the math problem, but name it "riddle"
* guess_correct: if player guessed correctly. The next state is "finished"
* guess_incorrect: if player guessed incorrectly. You should give player a clue. The next state is "guessing_riddle"
"""

history_concatenation = """Current conversation:
Player: ```{input}```
"""


class RoomLoopAnswer(BaseModel):
    """Room Loop Answer"""
    action: str = Field(
        description="The action to take. Must be one of the valid actions")
    similarity: float = Field(
        description="the similarity of the main idea of the player's answer to the main idea of the correct answer "
                    "as a value between 0 and 1")
    answer_idea: str = Field(
        description="the main idea of the player's answer")
    correct_answer_idea: str = Field(
        description="the main idea of the correct answer")
    reply: str = Field(
        description="Your reply to player. You must say clues in your own words instead of just copying them")
    new_state: str = Field(
        description="The new state, after this action. Must be one of the valid states")


RoomLoopAnswerParser = PydanticOutputParser(pydantic_object=RoomLoopAnswer)


def math_room_chain(topic: str = "Math", riddle: dict = None):
    langchain.llm_cache = InMemoryCache()

    model_name = "OpenAI"
    # model_name = "Replicate"
    # model_name = "Cohere"
    # model_name = "HuggingFace_google_flan"
    # model_name = "HuggingFace_mbzai_lamini_flan"
    # model_name = "Local_gpt2"
    # model_name = "Local_lama"
    llm = get_model(model_name, temperature=0.2)

    system_prompt = ADVENTURE_GAME_MATH_ROOM_PROMPT_TEMPLATE.format(
        topic=topic,
        riddle=riddle["question"],
        answer=riddle["answer"],
        actions=actions_str,
        states=states_str
    )

    template = system_prompt + history_concatenation + "\n{format_instructions}\n"

    prompt = PromptTemplate(
        template=template,
        input_variables=["input"],
        partial_variables={"format_instructions": RoomLoopAnswerParser.get_format_instructions()},
    )

    # We don't need memory in this case, although technically It's a conversation
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        # memory=memory,
        verbose=__DEBUG__
    )
    return conversation


def math_room_game_loop(topic: str = "Math"):
    # riddle = {
    #     "question": "What is the sum of 2 and 2?",
    #     "answer": "4"
    # }
    riddle = generate_math_riddle()
    print(f"Debug::: {riddle}")
    conversation = math_room_chain(topic=topic, riddle=riddle)

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


def get_mathe_question_generator():
    math_question_prompt = """
    You are a world class algorithm for generating math questions. 
    Your task is to generate a math question that requires calculation.

    Hint: The question should not contain the answer. 
    {input}
    """
    model_name = "Replicate"  # Provides more interesting questions
    # model_name = "OpenAI"   # ALso provides interesting questions
    # but now configured to provide debug simple questions
    q_llm = get_model(model_name, temperature=0.8)

    prompt = PromptTemplate(template=math_question_prompt, input_variables=["input"])
    q_chain = LLMChain(llm=q_llm, prompt=prompt, output_key="question", verbose=__DEBUG__)

    m_llm = get_model("OpenAI", temperature=0)
    m_chain = LLMMathChain.from_llm(llm=m_llm, verbose=__DEBUG__)

    overall_chain = SequentialChain(
        chains=[q_chain, m_chain],
        input_variables=["input"],
        output_variables=["question", "answer"],
        verbose=__DEBUG__
    )

    return overall_chain


def generate_math_riddle() -> dict[str, Any]:
    chain = get_mathe_question_generator()
    for _ in range(3):
        try:
            return chain({"input": "Generate a math question"})
        except Exception as e:
            print(e)
            continue
    raise ValueError("Can't parse the output")


if __name__ == '__main__':
    # r = generate_math_riddle()
    # print("Question:", r["question"])
    # print("Answer:", r["answer"])
    math_room_game_loop(topic="Math")
