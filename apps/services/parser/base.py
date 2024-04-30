
from abc import ABC

class BaseParser(ABC):
    def __init__(self, *args,**kwargs):
        pass

    def parse(self, *args, **kwargs):
        raise NotImplementedError("Please implement parse method!")
