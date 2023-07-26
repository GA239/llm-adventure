import logging

from dotenv import load_dotenv, find_dotenv
from langchain import ConversationChain

from adventure.utils import get_model, get_default_kwargs


def create_conversation_chain(model_name):
    llm = get_model(model_name, **get_default_kwargs(model_name))
    return ConversationChain(llm=llm, verbose=True)


def test_my_name(model_name):
    conversation = create_conversation_chain(model_name)
    logging.info(conversation.run("Hi!, my name is Andrei."))
    logging.info(conversation.run("What's my name? Answer in two words."))


def test_character_name(model_name):
    conversation = create_conversation_chain(model_name)
    logging.info(conversation.run("What the real name of Spider man?"))
    logging.info(conversation.run("What the real name of Batman? Answer in two words."))


def test_calculate(model_name):
    conversation = create_conversation_chain(model_name)
    prompt = "Calculate ```{}``` and tell me the result as a number. Be precise in calculation!"
    logging.info(conversation.run(prompt.format("2 + 2 * 3")))
    logging.info(conversation.run(prompt.format("(2 + 2) * 3")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    test_my_name("Replicate")
    test_character_name("Replicate")
    test_calculate("Replicate")

    logging.info("Smoke test passed!")
