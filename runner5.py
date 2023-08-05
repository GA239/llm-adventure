from typing import Any

from dotenv import load_dotenv, find_dotenv
from langchain import LLMChain, LLMMathChain, PromptTemplate
from langchain.chains import SequentialChain

from adventure.utils import get_model, get_default_kwargs

load_dotenv(find_dotenv(raise_error_if_not_found=True))


def get_riddle_generator():
    sys_prompt = """
    You are a world class algorithm for generating math questions. 
    Your task is to generate a math question that requires calculation.

    Hint: The question should not contain the answer. 
    {input}
    """
    model_name = "OpenAI"
    q_kwargs = {**get_default_kwargs(model_name), "temperature": 0.8}
    q_llm = get_model(model_name, **q_kwargs)

    prompt = PromptTemplate(template=sys_prompt, input_variables=["input"])
    q_chain = LLMChain(llm=q_llm, prompt=prompt, verbose=True, output_key="question")

    m_kwargs = {**get_default_kwargs(model_name), "temperature": 0}
    m_llm = get_model(model_name, **m_kwargs)
    m_chain = LLMMathChain.from_llm(llm=m_llm)

    overall_chain = SequentialChain(
        chains=[q_chain, m_chain],
        input_variables=["input"],
        output_variables=["question", "answer"],
        return_all=True,
        verbose=True)

    return overall_chain


def generate_riddle() -> dict[str, Any]:
    chain = get_riddle_generator()
    for _ in range(2):
        try:
            return chain({"input": "Generate a math question"})
        except Exception as e:
            print(e)
            continue
    raise ValueError("Can't parse the output")


r = generate_riddle()
print("Question:", r["question"])
print("Answer:", r["answer"])
