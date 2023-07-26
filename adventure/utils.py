import os
import openai
from langchain import OpenAI, HuggingFaceHub
from langchain.llms import Replicate, Cohere


def get_model(name, **kwargs):
    return {
        "OpenAI": get_openai_model,
        "Replicate": get_replicate_model,
        "Cohere": get_cohere_model,
        "HuggingFace": get_huggingface_model,
    }.get(name)(**kwargs)


def get_default_kwargs(name):
    return {
        "OpenAI": {},
        "Replicate": {"temperature": 0.1, "max_length": 100, "top_p": 1},
        "Cohere": {},
        "HuggingFace": {"temperature": 0.9, "max_length": 110},
    }.get(name)


def get_replicate_model(**kwargs):
    model_id = "a16z-infra/llama13b-v2-chat:6b4da803a2382c08868c5af10a523892f38e2de1aafb2ee55b020d9efef2fdb8"
    return Replicate(model=model_id, input=kwargs)


def get_openai_model(**kwargs):  # expensive
    openai.api_key = os.getenv('OPENAI_API_KEY')
    return OpenAI(**kwargs)


def get_cohere_model(**kwargs):  # API limitation: 5 calls per minute
    return Cohere(cohere_api_key=os.getenv('COHERE_API_KEY'), **kwargs)


def get_huggingface_model(**kwargs):
    repo_id = "google/flan-t5-xxl"  # not super clever
    return HuggingFaceHub(repo_id=repo_id, model_kwargs=kwargs)
