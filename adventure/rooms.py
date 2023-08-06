from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

import langchain
from langchain import LLMChain, LLMMathChain, PromptTemplate
from langchain.cache import InMemoryCache
from langchain.chains import SequentialChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from adventure.utils import get_model, game_print_debug, game_print
from adventure.utils import run_with_attempts, verbose


def get_room_by_type(room_type: str) -> Type[GeneralRoom] | Type[MathRoom] | None:
    return {
        "general": GeneralRoom,
        "math": MathRoom,
    }.get(room_type, None)


class Room(ABC):
    _riddle_generator: LLMChain = None
    """The riddle generator for this room. Generates riddle and answer"""

    _room_chain: LLMChain = None
    """The chain for this room that will be used to check 
    answers for the riddle. Isn't going to be a Conversation, 
    cause we actually don't need memory."""

    room_config: dict = None
    """The config for this room. Contains the topic and the room type"""

    _riddle: dict = None  # {"riddle": str, "answer": str}

    def __init__(self, room_config: dict):
        self.room_config = room_config

    @property
    def riddle(self):
        if not self._riddle:
            raise ValueError("Riddle is not generated yet")
        return self._riddle["riddle"]

    @property
    def answer(self):
        if not self._riddle:
            raise ValueError("answer is not generated yet")
        return self._riddle["answer"]

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

    @abstractmethod
    def _get_riddle_generator(self) -> LLMChain:
        raise NotImplementedError

    @property
    def riddle_generator(self) -> LLMChain:
        if not self._riddle_generator:
            self._riddle_generator = self._get_riddle_generator()
        return self._riddle_generator

    @abstractmethod
    def _get_room_chain(self) -> LLMChain:
        raise NotImplementedError

    @property
    def room_chain(self) -> LLMChain:
        if not self._room_chain:
            self._room_chain = self._get_room_chain()
        return self._room_chain

    @run_with_attempts
    def sub_loop(self, p_input):
        game_print_debug(f"chain input: {p_input}")
        repl = self.room_chain.run(p_input)
        game_print_debug(f"chain repl: {type(repl)} : {repl}")
        return self.RoomLoopAnswerParser.parse(repl).dict()

    def loop(self):
        game_print_debug("Debug::: Room Loop Started")

        rpl = self.sub_loop("Hello!, I'm here to solve the riddle.")
        game_print(f"Expert: {rpl['reply']}")

        while True:
            p_input = input("You: >>>")
            rpl = self.sub_loop(p_input)

            game_print(f"Expert: {rpl['reply']}")

            action, state = rpl["action"], rpl["new_state"]
            if state == "finished":
                break

            similarity = float(rpl["similarity"])
            if similarity > 0.65:
                game_print("However, Your guess is almost correct!")
                game_print(f"The answer is {self.answer}")
                break

        game_print("Thank you for playing!")


class GeneralRoom(Room):
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

    RiddleParser = PydanticOutputParser(pydantic_object=Riddle)

    def _get_riddle_generator(self) -> LLMChain:
        sys_prompt = """
        You are a world class algorithm for generating riddles. 
        Your task is to generate a riddle about {topic}.
        Your knowledge of {topic} should be used to generate riddle.

        Hint: The riddle should be short.
        Hint: The riddle should not contain the answer.
        """
        vbs = verbose()
        llm = get_model("OpenAI", temperature=0.8)

        prompt = PromptTemplate(
            template=sys_prompt + "\n{format_instructions}\n",
            input_variables=["topic"],
            partial_variables={"format_instructions": self.RiddleParser.get_format_instructions()},
        )
        return LLMChain(llm=llm, prompt=prompt, verbose=vbs)

    @run_with_attempts
    def generate_riddle(self):
        repl = self.riddle_generator.run(topic=self.room_config["topic"])
        game_print_debug(f"chain repl: {type(repl)} : {repl}")
        self._riddle = self.RiddleParser.parse(repl).dict()

    def _get_room_chain(self) -> LLMChain:
        self.generate_riddle()
        riddle = self._riddle
        game_print_debug(f"Debug ::: Riddle :: {riddle}")
        topic = self.room_config["topic"]
        return self._get_general_room_chain(riddle=riddle, topic=topic)

    def _get_general_room_chain(self, riddle: dict, topic: str) -> LLMChain:
        # TODO: check if this is needed
        langchain.llm_cache = InMemoryCache()
        vbs = verbose()

        model_name = "OpenAI"
        # model_name = "Replicate"
        # model_name = "Cohere"
        # model_name = "HuggingFace_google_flan"
        # model_name = "HuggingFace_mbzai_lamini_flan"
        # model_name = "Local_gpt2"
        # model_name = "Local_lama"
        llm = get_model(model_name, temperature=0.2)

        system_prompt = self.ADVENTURE_GAME_ROOM_PROMPT_TEMPLATE.format(
            topic=topic,
            riddle=riddle["riddle"],
            answer=riddle["answer"],
            actions=self.actions_str,
            states=self.states_str.format(topic=topic)
        )

        template = system_prompt + self.history_concatenation + "\n{format_instructions}\n"

        prompt = PromptTemplate(
            template=template,
            # input_variables=["history", "input"],
            input_variables=["input"],
            partial_variables={"format_instructions": self.RoomLoopAnswerParser.get_format_instructions()},
        )
        conversation = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=vbs,
        )
        return conversation


class MathRoom(Room):
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

    def _get_riddle_generator(self):
        math_question_prompt = """
        You are a world class algorithm for generating math questions. 
        Your task is to generate a math question that requires calculation.

        Hint: The question should not contain the answer. 
        {input}
        """
        vbs = verbose()
        # model_name = "Replicate"  # Provides more interesting questions
        model_name = "OpenAI"   # ALso provides interesting questions
        # but now configured to provide debug simple questions
        q_llm = get_model(model_name, temperature=0.8)

        prompt = PromptTemplate(template=math_question_prompt, input_variables=["input"])
        q_chain = LLMChain(llm=q_llm, prompt=prompt, output_key="question", verbose=vbs)

        m_llm = get_model("OpenAI", temperature=0)
        m_chain = LLMMathChain.from_llm(llm=m_llm, verbose=vbs)

        overall_chain = SequentialChain(
            chains=[q_chain, m_chain],
            input_variables=["input"],
            output_variables=["question", "answer"],
            verbose=vbs
        )

        return overall_chain

    @run_with_attempts
    def generate_riddle(self):
        repl = self.riddle_generator({"input": "Generate a math question"})
        game_print_debug(f"chain repl: {type(repl)} : {repl}")
        self._riddle = repl

    def _get_room_chain(self) -> LLMChain:
        self.generate_riddle()
        riddle = self._riddle
        game_print_debug(f"Debug ::: Riddle :: {riddle}")
        return self._get_math_room_chain(riddle=riddle)

    def _get_math_room_chain(self, riddle: dict, topic: str = "Math") -> LLMChain:
        # TODO: check if this is needed
        langchain.llm_cache = InMemoryCache()
        vbs = verbose()

        model_name = "OpenAI"
        # model_name = "Replicate"
        # model_name = "Cohere"
        # model_name = "HuggingFace_google_flan"
        # model_name = "HuggingFace_mbzai_lamini_flan"
        # model_name = "Local_gpt2"
        # model_name = "Local_lama"
        llm = get_model(model_name, temperature=0.2)

        system_prompt = self.ADVENTURE_GAME_MATH_ROOM_PROMPT_TEMPLATE.format(
            topic=topic,
            riddle=riddle["question"],
            answer=riddle["answer"],
            actions=self.actions_str,
            states=self.states_str
        )

        template = system_prompt + self.history_concatenation + "\n{format_instructions}\n"

        prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            partial_variables={"format_instructions": self.RoomLoopAnswerParser.get_format_instructions()},
        )

        # We don't need memory in this case, although technically It's a conversation
        conversation = LLMChain(
            llm=llm,
            prompt=prompt,
            # memory=memory,
            verbose=vbs
        )
        return conversation
