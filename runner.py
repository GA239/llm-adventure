import logging

from dotenv import load_dotenv, find_dotenv
from langchain import ConversationChain

from adventure.utils import get_model, get_default_kwargs

text = f"""
You should express what you want a model to do by 
providing instructions that are as clear and 
specific as you can possibly make them. 
This will guide the model towards the desired output,  
and reduce the chances of receiving irrelevant 
or incorrect responses. Don't confuse writing a  
clear prompt with writing a short prompt. 
In many cases, longer prompts provide more clarity  
and context for the model, which can lead to 
more detailed and relevant outputs.
"""

prompt = f"""
Summarize the text delimited by triple backticks 
into a single sentence, ten words.
```{text}```
"""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    llm = get_model("Replicate", **get_default_kwargs("Replicate"))

    conversation = ConversationChain(llm=llm, verbose=True)
    logging.info(conversation.run(prompt))
