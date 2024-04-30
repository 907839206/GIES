
import logging,traceback
from abc import ABC,abstractmethod

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseModel(ABC):

    def __init__(self,*args,**kwargs):
        try:
            self.load_model(*args,**kwargs)
            logger.info(f"load model success")
        except Exception as _:
            logger.error(f"load model error, msg:{traceback.format_exc()}")

    @abstractmethod
    def load_model(self,*args,**kwargs):
        pass

    @abstractmethod
    def chat(self,*args,**kwargs):
        pass

    @abstractmethod
    def encode(self,*args,**kwargs):
        pass

    @abstractmethod
    def encode_queries(self,*args,**kwargs):
        pass

    @abstractmethod
    def describe(self,*args,**kwargs):
        pass




