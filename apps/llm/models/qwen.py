
import logging
from .base import BaseModel
from dashscope import Generation
from http import HTTPStatus

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Qwen(BaseModel):

    def __init__(self,api_key = None,
                model_name = Generation.Models.qwen_turbo,
                **kwargs):
        if not api_key:
            assert ValueError("api_key can not be None!")
        super().__init__(**kwargs)
        import dashscope
        dashscope.api_key = api_key
        self.model_name = model_name
        self.temperature_d = 0.1
        self.top_p_d = 0.3
        self.frequency_penalty_d = 0.7
        self.presence_penalty_d = 0.4
        self.max_tokens_d = 1024 # 仅215？

    def __chat_param_update(self,gen_conf):
        if not gen_conf.get("temperature",None):
            gen_conf.update({"temperature":self.temperature_d})
        if not gen_conf.get("top_p",None):
            gen_conf.update({"top_p":self.top_p_d})
        if not gen_conf.get("frequency_penalty",None):
            gen_conf.update({"frequency_penalty":self.frequency_penalty_d})
        if not gen_conf.get("presence_penalty",None):
            gen_conf.update({"presence_penalty":self.presence_penalty_d})
        if not gen_conf.get("max_tokens",None):
            gen_conf.update({"max_tokens":self.max_tokens_d})


    def load_model(self,*args,**kwargs):
        logger.info("external model, doesn't need to initilize!")

    def chat(self, system, message, gen_conf):
        if system:
            message.insert(0, {"role": "system", "content": system})
        self.__chat_param_update(gen_conf)
        response = Generation.call(
            self.model_name,
            messages=message,
            result_format='message',
            **gen_conf
        )
        ans = ""
        tk_count = 0
        if response.status_code == HTTPStatus.OK:
            ans += response.output.choices[0]['message']['content']
            tk_count += response.usage.total_tokens
            if response.output.choices[0].get("finish_reason", "") == "length":
                ans += "\n由于长度的原因，回答被截断了，要继续吗？"
            return ans, tk_count
        return "**ERROR**: " + response.message, tk_count


    def describe(self,*args,**kwargs):
        pass

    def embd_preprocess(self,infos):
        new_infos = []
        for info in infos:
            # create ES index
            if not info.get("_id",None):
                md5 = hashlib.md5()
                md5.update(info["content_with_weight"].encode("utf-8"))
                info["_id"] = md5.hexdigest()
            if not info.get("create_time",None):
                info["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
            if not info.get("create_timestamp_flt",None):
                info["create_timestamp_flt"] = datetime.datetime.now().timestamp()
            new_infos.append(info)
        return new_infos

    def encode(self, system, message: list, gen_conf):
        batch_size = gen_conf.get("batch_size", 10)
        res = []
        token_count = 0
        texts = [txt[:2048] for txt in message]
        for i in range(0, len(texts), batch_size):
            resp = dashscope.TextEmbedding.call(
                model=self.model_name,
                input=texts[i:i + batch_size],
                text_type="document"
            )
            embds = [[] for _ in range(len(resp["output"]["embeddings"]))]
            for e in resp["output"]["embeddings"]:
                embds[e["text_index"]] = e["embedding"]
            res.extend(embds)
            token_count += resp["usage"]["total_tokens"]
        return np.array(res), token_count

    def encode_queries(self, text):
        resp = dashscope.TextEmbedding.call(
            model=self.model_name,
            input=text[:2048],
            text_type="query"
        )
        return np.array(resp["output"]["embeddings"][0]
                        ["embedding"]), resp["usage"]["total_tokens"]
