import sys,os
# sys.path.append(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         "../../"
#     )
# )

import cv2
import numpy as np
from PIL import Image

from services.ocr import load_model,OCR
from services.ocr.operators import *
from setting import config
from services.utils import get_project_path




def distance(box_1, box_2):
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    dis = abs(x3 - x1) + abs(y3 - y1) + abs(x4 - x2) + abs(y4 - y2)
    dis_2 = abs(x3 - x1) + abs(y3 - y1)
    dis_3 = abs(x4 - x2) + abs(y4 - y2)
    return dis + min(dis_2, dis_3)


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


class Tsr:
    def __init__(self,model_dir=None):
        assert model_dir != None,"model dir can not be None!"
        self.model_dir = model_dir
        self.predictor,_ = load_model(model_dir,"ch_pp_tsr")
        self.character = self.__get_meta()
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

    def __get_meta(self):
        meta_dict = self.predictor.get_modelmeta().custom_metadata_map
        content_list = meta_dict["character"].splitlines()
        return content_list

    def __get_input_names(self):
        return [v.name for v in self.predictor.get_inputs()]
    
    def __get_output_names(self):
        return [v.name for v in self.predictor.get_outputs()]

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
        for operator in self.preprocess_list:
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

    def __call__(self,img):
       
        if isinstance(img,Image.Image):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        data = {"image":img}
        data = self.preprocess(data)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()

        input_dict = dict(zip(self.__get_input_names(), [img]))
        outputs = self.predictor.run(self.__get_output_names(), input_dict)

        preds = {"loc_preds": outputs[0], "structure_probs": outputs[1]}
        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess(preds, [shape_list])

        bbox_list = post_result["bbox_batch_list"][0]
        structure_str_list = post_result["structure_batch_list"][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = (
            ["<html>", "<body>", "<table>"]
            + structure_str_list
            + ["</table>", "</body>", "</html>"]
        )
        return structure_str_list,bbox_list,{"h":h,"w":w}

    def match_ocr(self,pred_structure,pred_bbox,dt_bbox,ocr_res):

        def __filter_ocr_res(pred_bbox,dt_bbox,ocr_res):
            y1 = pred_bbox[:, 1::2].min()
            new_dt_boxes = []
            new_rec_res = []
            for box, rec in zip(dt_bbox, ocr_res):
                if np.max(box[1::2]) < y1:
                    continue
                new_dt_boxes.append(box)
                new_rec_res.append(rec)
            return new_dt_boxes, new_rec_res

        def __match_ret(dt_boxes,pred_bboxes):
            matched = {}
            for i, gt_box in enumerate(dt_boxes):
                distances = []
                for j, pred_box in enumerate(pred_bboxes):
                    if len(pred_box) == 8:
                        pred_box = [
                            np.min(pred_box[0::2]),
                            np.min(pred_box[1::2]),
                            np.max(pred_box[0::2]),
                            np.max(pred_box[1::2]),
                        ]
                    distances.append(
                        (distance(gt_box, pred_box), 1.0 - compute_iou(gt_box, pred_box))
                    )  # compute iou and l1 distance
                sorted_distances = distances.copy()
                # select det box by iou and l1 distance
                sorted_distances = sorted(
                    sorted_distances, key=lambda item: (item[1], item[0])
                )
                if distances.index(sorted_distances[0]) not in matched.keys():
                    matched[distances.index(sorted_distances[0])] = [i]
                else:
                    matched[distances.index(sorted_distances[0])].append(i)
            return matched
        
        def __get_pred_html(pred_structures, matched_index, ocr_contents):
            end_html = []
            td_index = 0
            for tag in pred_structures:
                if "</td>" not in tag:
                    end_html.append(tag)
                    continue
                if "<td></td>" == tag:
                    end_html.extend("<td>")
                if td_index in matched_index.keys():
                    b_with = False
                    if (
                        "<b>" in ocr_contents[matched_index[td_index][0]]
                        and len(matched_index[td_index]) > 1
                    ):
                        b_with = True
                        end_html.extend("<b>")
                    for i, td_index_index in enumerate(matched_index[td_index]):
                        content = ocr_contents[td_index_index][0]
                        if len(matched_index[td_index]) > 1:
                            if len(content) == 0:
                                continue
                            if content[0] == " ":
                                content = content[1:]
                            if "<b>" in content:
                                content = content[3:]
                            if "</b>" in content:
                                content = content[:-4]
                            if len(content) == 0:
                                continue
                            if i != len(matched_index[td_index]) - 1 and " " != content[-1]:
                                content += " "
                        end_html.extend(content)
                    if b_with:
                        end_html.extend("</b>")
                if "<td></td>" == tag:
                    end_html.append("</td>")
                else:
                    end_html.append(tag)
                td_index += 1
            # Filter <thead></thead><tbody></tbody> elements
            filter_elements = ["<thead>", "</thead>", "<tbody>", "</tbody>"]
            end_html = [v for v in end_html if v not in filter_elements]
            return "".join(end_html), end_html
    
        dt_boxes, rec_res = __filter_ocr_res(pred_bbox, dt_bbox, ocr_res)
        matched_index = __match_ret(dt_boxes, pred_bbox)
        pred_html,_=__get_pred_html(pred_structure,matched_index,rec_res)
        return pred_html

def get_boxes_recs(ocr_result, h, w) :
    def __ocr_res_adaptor(ocr_res):
        new_ocr_res = []
        for i in range(len(ocr_res)):
            new_data = []
            data = ocr_res[i]
            new_data.append(data[0])
            for d in data[1]:
                new_data.append(d)
            new_ocr_res.append(new_data)
        return new_ocr_res
    
    ocr_result = __ocr_res_adaptor(ocr_result)
    dt_boxes, rec_res, scores = list(zip(*ocr_result))
    rec_res = list(zip(rec_res, scores))
    r_boxes = []
    for box in dt_boxes:
        box = np.array(box)
        x_min = max(0, box[:, 0].min() - 1)
        x_max = min(w, box[:, 0].max() + 1)
        y_min = max(0, box[:, 1].min() - 1)
        y_max = min(h, box[:, 1].max() + 1)
        box = [x_min, y_min, x_max, y_max]
        r_boxes.append(box)
    dt_boxes = np.array(r_boxes)
    return dt_boxes, rec_res


if __name__=="__main__":
    filepath = "/workspaces/GIES/static/order_table.jpeg"
    _model_path = os.path.join(get_project_path(),config.OCR.MODEL_PATH)
    model = Tsr(_model_path)
    img = Image.open(filepath)
    pred_structures, pred_bboxes,wh = model(img)
    ocr_model = OCR(_model_path)
    ocr_ret = ocr_model(np.array(img.convert('RGB')))
    dt_boxes, rec_res = get_boxes_recs(ocr_ret, wh.get("h"), wh.get("w"))
    pred_html = model.match_ocr(pred_structures, pred_bboxes, dt_boxes, rec_res)
    print(f"pred_html:{pred_html}")
    with open("save.html",'w') as f:
        f.write(pred_html)

