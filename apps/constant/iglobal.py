
import os

from services.parser import PictureParser
# from components.isearch import ElasticSearch

from setting import config
from llm.models import LLMDrawer
from llm import prompt_default


def _model_defalut(config,llm_type=None):
    if llm_type == "chat":
        factory_default = config.CHAT.DEFAULT.FACTORY
        model_default = config.CHAT.DEFAULT.NAME
        API_KEY = config.CHAT.DEFAULT.API_KEY
    elif llm_type == "embd":
        factory_default = config.EMBEDDING.DEFAULT.FACTORY
        model_default = config.EMBEDDING.DEFAULT.NAME
        API_KEY = config.EMBEDDING.DEFAULT.API_KEY
    else:
        raise ValueError(f"got unexpected llm_type:{llm_type}")
    assert API_KEY != None, "Default API KEY must be set!"
    if factory_default not in LLMDrawer.keys():
        raise ValueError(f"{factory_default} not support!")
    llm_model = LLMDrawer[factory_default](API_KEY,model_default)
    return llm_model

def _model_init(factory,apikey,base_url):
    llm_model = LLMDrawer[factory](apikey,base_url)
    return llm_model 

class IGlobal:
    pic_parse = PictureParser()
    chat_model = _model_defalut(config,llm_type = "chat")
    embd_model = _model_defalut(config,llm_type = "embd")
    prompt_default = prompt_default
    model_list = vars(config.CHAT)
    supported_model = {}
    supported_model_mapping = {}
    for model_key,model_info in model_list.items():
        if model_key.lower() == "default":
            continue
        _model_ist = _model_init(model_info.FACTORY,model_info.API_KEY,model_info.BASE_URL)
        supported_model[model_key.lower()] = _model_ist
        supported_model_mapping[model_key.lower()] = model_info.NAME


iglobal = IGlobal()