import sys,os
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../"
    )
)


import numpy as np
from PIL import Image

from services.ocr import load_model
from services.ocr.operators import *
from setting import config
from services.utils import get_project_path


class Tsr:
    def __init__(self,model_dir=None):
        assert model_dir != None,"model dir can not be None!"
        self.model_dir = model_dir
        self.predictor,_ = load_model(model_dir,"tsr")
        self.character = self.predictor.get_metadata()
        self.init()

        self.table_max_len = 488
        self.preprocess_list = self.__build_pre_process_list()
        self.preprocess_op_list = self.__build_preprocess_op_list()

    def init(self):
        if "<td></td>" not in self.character:
            self.character.append("<td></td>")
        if "<td>" in self.character:
            self.character.remove("<td>")
        self.beg_str = "sos"
        self.end_str = "eos"
        self.character = [self.beg_str] + self.character + [self.end_str]
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i
        self.td_token = ["<td>", "<td", "<td></td>"]

    def __build_pre_process_list(self):
        resize_op = {
            "ResizeTableImage": {
                "max_len": self.table_max_len,
            }
        }
        pad_op = {
            "PaddingTableImage": {"size": [self.table_max_len, self.table_max_len]}
        }
        normalize_op = {
            "NormalizeImage": {
                "std": [0.229, 0.224, 0.225],
                "mean": [0.485, 0.456, 0.406],
                "scale": "1./255.",
                "order": "hwc",
            }
        }
        to_chw_op = {"ToCHWImage": None}
        keep_keys_op = {"KeepKeys": {"keep_keys": ["image", "shape"]}}
        return [resize_op, normalize_op,pad_op, to_chw_op, keep_keys_op]

    def __build_preprocess_op_list(self):
        ops = []
        for operator in self.pre_process_list:
            assert (
                isinstance(operator, dict) and len(operator) == 1
            ), "yaml format error"
            op_name = list(operator)[0]
            param = {} if operator[op_name] is None else operator[op_name]
            op = eval(op_name)(**param)
            ops.append(op)
        return ops

    def preprocess(self,data):
        for op in self.preprocess_op_list:
            data = op(data)
            if data is None:
                return None
        return data

    def postprocess(self,preds,batch=None):
        structure_probs = preds["structure_probs"]
        bbox_preds = preds["loc_preds"]
        shape_list = batch[-1]
        result = self.__decode(structure_probs, bbox_preds, shape_list)
        if len(batch) == 1:
            return result

    def __decode(self,structure_probs, bbox_preds, shape_list):
        ignored_tokens = self.__get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            score_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break

                if char_idx in ignored_tokens:
                    continue

                text = self.character[char_idx]
                if text in self.td_token:
                    bbox = bbox_preds[batch_idx, idx]
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])
            structure_batch_list.append([structure_list, np.mean(score_list)])
            bbox_batch_list.append(np.array(bbox_list))
        result = {
            "bbox_batch_list": bbox_batch_list,
            "structure_batch_list": structure_batch_list,
        }
        return result

    def _bbox_decode(self, bbox, shape):
        h, w = shape[:2]
        bbox[0::2] *= w
        bbox[1::2] *= h
        return bbox

    def __get_ignored_tokens(self):
        beg_idx = np.array(self.dict[self.beg_str])
        end_idx = np.array(self.dict[self.end_str])
        return [beg_idx, end_idx]

    def __call__(self,image_list):
        for _,img in enumerate(image_list):
            h, w = img.shape[:2]
            print(f" ----> h:{h}  w:{w}")
            data = {"image":img}
            data = self.preprocess(data)
            img = data[0]
            if img is None:
                return None, 0
            img = np.expand_dims(img, axis=0)
            img = img.copy()
            outputs = self.predictor(img)
            preds = {"loc_preds": outputs[0], "structure_probs": outputs[1]}
            shape_list = np.expand_dims(data[-1], axis=0)
            post_result = self.postprocess(preds, [shape_list])
            return post_result


if __name__=="__main__":
    filepath = "/workspaces/GIES/static/order1.jpg"
    _model_path = os.path.join(get_project_path(),config.OCR.MODEL_PATH)
    model = Tsr(_model_path)
    image_list = [
        Image.open(filepath)
    ]
    res_list = model(image_list)

