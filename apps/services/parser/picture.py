import io,logging,os,traceback
from enum import Enum

import requests
from PIL import Image
import numpy as np

from setting import config
from services.ocr import OCR
from services.layout import LayoutRecognize
from services.tsr import Tsr,get_boxes_recs

from services.utils import get_project_path
from services.tokenizer import tokenize
from constant import LayoutType

from .base import BaseParser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PictureParser(BaseParser):

    def __init__(self):
        _model_path = os.path.join(get_project_path(),config.OCR.MODEL_PATH)
        self.__ocr = OCR(_model_path)
        self.__layout = LayoutRecognize(_model_path)
        self.__tsr = Tsr(_model_path)
        self.__threshold = 0.2

    def __decode_local(self,path):
        if not os.path.isfile(path):
            logger.error(f"{path} is not a valide path!")
            return None
        _img = Image.open(path).convert('RGB')
        return _img

    def __check_layout(self,layout_res):
        for idx,res in enumerate(layout_res):
            for _,info in enumerate(res):
                if info["type"] =="table" and info["score"] > self.__threshold:
                    return True
        return False
    

    def parse(self,img_path,layout_type):
        try:
            _img = self.__decode_local(img_path)
            if not _img:
                return []
            if layout_type == LayoutType.general.value:
                # (1) 版面识别
                _img_list = [_img]
                _res_list = self.__layout(_img_list,thr=self.__threshold)
                print(f"[DEBUG] _res_list:{_res_list}")
                _use_tsr = self.__check_layout(_res_list)
                _mask_img_list = self.__layout.mask_entity(_img_list,_res_list,self.__threshold)
                # (2) 表格识别
                _table_ret = []
                if _use_tsr:
                    # TSR
                    print("[DEBUG] start using tsr...")
                    _table_img_list = self.__layout.crop_tables(_img_list,_res_list,self.__threshold)[0]
                    for _table_img in _table_img_list:
                        _pred_structures, _pred_bboxes,_wh = self.__tsr(_table_img)
                        _table_ocr_ret = self.__ocr(np.array(_table_img.convert('RGB')))
                        if not _table_ocr_ret or len(_table_ocr_ret)==0:
                            continue
                        _dt_boxes, _rec_res = get_boxes_recs(_table_ocr_ret, _wh.get("h"), _wh.get("w"))
                        _pred_html = self.__tsr.match_ocr(_pred_structures, _pred_bboxes, _dt_boxes, _rec_res)
                        _table_ret.append(_pred_html)
                    print(f"[DEBUG] tsr result:{_table_ret}")
                # (3) 文字识别
                _msk_img = _mask_img_list[0]
                ret = {}
                _bxs = self.__ocr(np.array(_msk_img))
                _txt = "\n".join([t[0] for _, t in _bxs if t[0]])
                if _use_tsr and len(_table_ret) > 0: 
                    _txt += "\n".join(_table_ret)
                tokenize(ret, _txt)
                return [ret]
            elif layout_type == LayoutType.text.value:
                # (1) 文字识别
                ret = {}
                _bxs = self.__ocr(np.array(_img))
                _txt = "\n".join([t[0] for _, t in _bxs if t[0]])
                tokenize(ret, _txt)
                return [ret]

            elif layout_type == LayoutType.table.value:
                # (1) 表格识别
                _pred_structures, _pred_bboxes,_wh = self.__tsr(_img)
                _table_ocr_ret = self.__ocr(np.array(_img.convert('RGB')))
                if not _table_ocr_ret or len(_table_ocr_ret)==0:
                    return []
                _dt_boxes, _rec_res = get_boxes_recs(_table_ocr_ret, _wh.get("h"), _wh.get("w"))
                ret = {}
                _pred_html = self.__tsr.match_ocr(_pred_structures, _pred_bboxes, _dt_boxes, _rec_res)
                tokenize(ret, _pred_html)
                return [ret]
            else:
                raise ValueError("unsupported layout type!")
            # Layout
            # _img_list = [_img]
            # _res_list = self.__layout(_img_list,thr=self.__threshold)
            # print(f"[DEBUG] _res_list:{_res_list}")
            # _use_tsr = self.__check_layout(_res_list)
            # _mask_img_list = self.__layout.mask_entity(_img_list,_res_list,self.__threshold)

            # _table_ret = []
            # if _use_tsr:
            #     # TSR
            #     print("[DEBUG] start using tsr...")
            #     _table_img_list = self.__layout.crop_tables(_img_list,_res_list,self.__threshold)[0]
            #     for _table_img in _table_img_list:
            #         _pred_structures, _pred_bboxes,_wh = self.__tsr(_table_img)
            #         _table_ocr_ret = self.__ocr(np.array(_table_img.convert('RGB')))
            #         if not _table_ocr_ret or len(_table_ocr_ret)==0:
            #             continue
            #         _dt_boxes, _rec_res = get_boxes_recs(_table_ocr_ret, _wh.get("h"), _wh.get("w"))
            #         _pred_html = self.__tsr.match_ocr(_pred_structures, _pred_bboxes, _dt_boxes, _rec_res)
            #         _table_ret.append(_pred_html)
            #     print(f"[DEBUG] tsr result:{_table_ret}")
            
            # # OCR
            # _msk_img = _mask_img_list[0]
            # ret = {}
            # _bxs = self.__ocr(np.array(_msk_img))
            # _txt = "\n".join([t[0] for _, t in _bxs if t[0]])
            # if _use_tsr and len(_table_ret) > 0: 
            #     _txt += "\n".join(_table_ret)

            # tokenize(ret, _txt)
            # return [ret]
        except Exception as _:
            logger.error(f"[PictureParser][parse] err, msg:{traceback.format_exc()}")
            return []


        


        