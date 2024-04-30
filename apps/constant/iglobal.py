
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
        API_KEY = os.environ.get("API_KEY_CHAT",None)
    elif llm_type == "embd":
        factory_default = config.EMBEDDING.DEFAULT.FACTORY
        model_default = config.EMBEDDING.DEFAULT.NAME
        API_KEY = os.environ.get("API_KEY_EMBEDDING",None)
    else:
        raise ValueError(f"got unexpected llm_type:{llm_type}")
    assert API_KEY != None, "Default API KEY must be set!"
    if factory_default not in LLMDrawer.keys():
        raise ValueError(f"{factory_default} not support!")
    llm_model = LLMDrawer[factory_default](API_KEY,model_default)
    return llm_model


class IGlobal:
    pic_parse = PictureParser()
    chat_model = _model_defalut(config,llm_type = "chat")
    embd_model = _model_defalut(config,llm_type = "embd")
    # es_client = ElasticSearch()
    prompt_default = prompt_default


iglobal = IGlobal()