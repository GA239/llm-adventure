from typing import Any

from dotenv import load_dotenv, find_dotenv
from langchain import LLMChain, LLMMathChain, PromptTemplate
from langchain.chains import SequentialChain

from adventure.utils import get_model

load_dotenv(find_dotenv(raise_error_if_not_found=True))
__DEBUG__ = True


def get_mathe_question_generator():
    math_question_prompt = """
    You are a world class algorithm for generating math questions. 
    Your task is to generate a math question that requires calculation.

    Hint: The question should not contain the answer. 
    {input}
    """
    q_llm = get_model("OpenAI", temperature=0.8)

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


def generate_riddle() -> dict[str, Any]:
    chain = get_mathe_question_generator()
    for _ in range(3):
        try:
            return chain({"input": "Generate a math question"})
        except Exception as e:
            print(e)
            continue
    raise ValueError("Can't parse the output")


r = generate_riddle()
print("Question:", r["question"])
print("Answer:", r["answer"])
