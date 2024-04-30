import io,logging,os,traceback
from enum import Enum

import requests
from PIL import Image
import numpy as np

from setting import config
from services.ocr import OCR
from services.utils import get_project_path
from services.tokenizer import tokenize

from .base import BaseParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PictureParser(BaseParser):

    def __init__(self):
        _model_path = os.path.join(get_project_path(),config.OCR.MODEL_PATH)
        self.__ocr = OCR(_model_path)

    def __decode_local(self,path):
        if not os.path.isfile(path):
            logger.error(f"{path} is not a valide path!")
            return None
        _img = Image.open(path).convert('RGB')
        return _img

    def parse(self,img_path):
        try:
            _img = self.__decode_local(img_path)
            if not _img:
                return []
            ret = {}
            _bxs = self.__ocr(np.array(_img))
            _txt = "\n".join([t[0] for _, t in _bxs if t[0]])
            tokenize(ret, _txt)
            return [ret]
        except Exception as _:
            logger.error(f"[PictureParser][parse] err, msg:{traceback.format_exc()}")
            return []


        


        