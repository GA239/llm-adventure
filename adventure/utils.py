import os
from functools import partial

import openai
import torch
from langchain import OpenAI, HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers, HuggingFacePipeline, Replicate, Cohere
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaTokenizer, LlamaForCausalLM, pipeline, BitsAndBytesConfig


def get_model(name, **kwargs):
    return {
        "OpenAI": get_openai_model,
        "Replicate": get_replicate_model,
        "Cohere": get_cohere_model,
        "HuggingFace_google_flan": get_google_flan_t5_xxl,
        "HuggingFace_mbzai_lamini_flan": get_mbzai_lamini_flan_t5_783m,
        "Local_gpt2": get_local_gpt2_model,
        "Local_lama": get_local_lama_model,
    }.get(name)(**kwargs)


def get_default_kwargs(name):
    return {
        "OpenAI": {"temperature": 0.9},
        "Replicate": {"temperature": 0.7, "max_length": 100, "top_p": 1},
        "HuggingFace_google_flan": {"temperature": 0.5, "max_length": 1000},
        "HuggingFace_mbzai_lamini_flan": {"max_length": 512, "temperature": 0.7},
        "Local_lama": {'max_new_tokens': 32, 'repetition_penalty': 3.0, "temperature": 0.6},
    }.get(name, {})


def get_huggingface_model(repo_id, **kwargs):
    return HuggingFaceHub(repo_id=repo_id, model_kwargs=kwargs)


get_google_flan_t5_xxl = partial(get_huggingface_model, repo_id="google/flan-t5-xxl")
get_mbzai_lamini_flan_t5_783m = partial(get_huggingface_model, repo_id="MBZUAI/LaMini-Flan-T5-783M")


def get_replicate_model(**kwargs):
    # https://github.com/a16z-infra/cog-llama-template
    model_id = "a16z-infra/llama13b-v2-chat:d5da4236b006f967ceb7da037be9cfc3924b20d21fed88e1e94f19d56e2d3111"
    return Replicate(model=model_id, input=kwargs)


def get_openai_model(**kwargs):  # expensive
    openai.api_key = os.getenv('OPENAI_API_KEY')
    # return OpenAI(**kwargs)
    return ChatOpenAI(**kwargs)


def get_cohere_model(**kwargs):  # API limitation: 5 calls per minute
    return Cohere(cohere_api_key=os.getenv('COHERE_API_KEY'), **kwargs)


def get_local_model_path(model_name):
    # path = "../../models/"
    path = "./models/"
    model_path = os.path.join(path, model_name)
    return model_path


def get_local_gpt2_model(**kwargs):
    model_path = get_local_model_path("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    llm = GPT2LMHeadModel.from_pretrained(model_path)

    pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        # temperature=0.1,
        max_new_tokens=30,
        **kwargs
    )
    return HuggingFacePipeline(pipeline=pipe)


def get_local_alpaca_model(**kwargs):
    model_path = get_local_model_path("alpaca-native")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    base_model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        torch_dtype=torch.float32,
        device_map='auto',
        quantization_config=quantization_config,
        offload_folder="./offload",
    )
    pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2,
        **kwargs
    )
    return HuggingFacePipeline(pipeline=pipe)


def get_local_lama_model(**kwargs):
    model_path = get_local_model_path("Llama-2-7B-Chat-GGML")
    return CTransformers(model=model_path, model_type='llama', config=kwargs)
