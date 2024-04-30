

from abc import ABC,abstractmethod

class BaseHandler(ABC):
    def __init__(self, *args,**kwargs):
        pass
    
    @abstractmethod
    def handler(self, *args, **kwargs):
        raise NotImplementedError("Please implement handler method!")
