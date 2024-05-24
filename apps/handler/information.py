import os,logging,re,time

from elasticsearch_dsl import Q

from .base import BaseHandler
from constant import CodeEnum,iglobal,LayoutType
from llm import ModelService


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class InformationHandler(BaseHandler):

    llm_service = ModelService()

    @staticmethod
    def postprocess(answer_text):
        pattern = r'^```json([\s\S]*?)(```|\Z)'
        match = re.match(pattern, answer_text)
        if match:
            json_text = match.group(1)
            return json_text
        else:
            return answer_text

    @staticmethod
    def extract_markdown(answer_text):
        pattern = '```json\n[\{|\[](?:\n|.)+[\}|\]]\n```'
        result = re.search(pattern, answer_text)
        if result:
            matched_json = result.group()
            return matched_json
        else:
            return answer_text


    @classmethod
    def handler(cls,request):
        request_dict = request.get_json()
        fid = request_dict.get("fid",None)
        fpth = request_dict.get("fpth",None)
        extract_fields_str = request_dict.get("extract_fields",None)
        layout_type = request_dict.get("layout_type",LayoutType.general.value)
        if not fid or not fpth or not os.path.isfile(fpth):
            return {"ec":CodeEnum.Fail,"em":"file not exists","data":{}}
        _st = time.time()
        infos = iglobal.pic_parse.parse(fpth,layout_type)
        print(f"[INFO] ocr time cost:{time.time()-_st}")
        for e in infos:
            e.update({"fid":fid})
        print(f"[INFO] ocr result:{infos}")
        
        # TODO: 保存特征；相似性检索
        # tk_count = cls.llm_service.embedding(infos,iglobal.embd_model,parser_config = {})
        # index_name = iglobal.es_client.init_index(request["fid"])
        # chunk_count = len(set([info["_id"] for info in infos]))
        # es_r = iglobal.es_client.bulk(infos, index_name)
        # if es_r:
        #     iglobal.es_client.deleteByQuery(
        #         Q("match", fid=fid), idxnm=index_name)
        #     logger.error(f"[InformationHandler][handler] index fail!")
        # else:
        #     # 记录文档已使用token数和分区数
        #     logger.info(f"{fid} has token:{tk_count} chunk_count:{chunk_count}")
       
        # TODO：提取embd进行相似性检索
        
        # 抽取字段
        print(f"extract_fields_str:{extract_fields_str}")
        extract_fields_list = re.split(r'[;；]', extract_fields_str)
        extract_fields_list = [
            field.strip() for field in extract_fields_list if field.strip() !=""
        ]
        # 请求LLM
        knowledge = "\n".join([info.get("content_with_weight") for info in infos])
        default_prompt = iglobal.prompt_default.InformationExtractPrompt
        msg = [{
            "role": "user", 
            "content": default_prompt["querylogue"]. \
                format(fields = extract_fields_list)
        }]
        system = default_prompt["system"].format(knowledge = knowledge)
        gen_conf = {
            "temperature": request_dict.get("temperature"),
            "top_p": request_dict.get("top_p"),
            "frequency_penalty": request_dict.get("frequency_penalty"),
            "presence_penalty": request_dict.get("presence_penalty"),
            "max_tokens": request_dict.get("max_tokens")
        }
        _st = time.time()
        ans = cls.llm_service.chat(iglobal.chat_model,system, msg, **gen_conf)
        print(f"[DEBUG] ans:{ans[0]}")
        print(f"[INFO] llm time cost:{time.time()-_st}")
        ans = (cls.extract_markdown(ans[0]),cls.postprocess(ans[0]),ans[1])
        return ans
        


