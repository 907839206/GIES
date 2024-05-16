import copy

from openai import OpenAI

from .base import BaseModel


class Openai(BaseModel):

    def __init__(self, api_key = None, base_url = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        
        self.__client =  OpenAI(
            api_key = api_key,
            base_url = base_url,
        )

    def load_model(self,*args,**kwargs):
        pass


    def chat(self,system,messages,gen_conf):
        messages = copy.deepcopy(messages)
        if system:
            messages.insert(0, {"role": "system", "content": system})
        response = self.__client.chat.completions.create(
            messages = messages,
            **gen_conf
        )
        return response


    def encode(self,*args,**kwargs):
        pass


    def encode_queries(self,*args,**kwargs):
        pass


    def describe(self,*args,**kwargs):
        pass


    