
import re
import numpy as np

from services.utils import rmSpace


class ModelService:

    def __init__(self,*args,**kwargs):
        pass

    # TODO parser_config 应该是参数设置
    # 从DocumentService -> KnowledgebaseService来的，用户上传时的设定
    def embedding(self, docs, emb_model, parser_config={}):
        # 预处理
        prefix_method = ["preprocess"]
        for _method in prefix_method:
            if (hasattr(emb_model,_method) 
                    and callable(getattr(emb_model,_method))):
                call = getattr(emb_model, _method)
                docs = call(docs)

        # TODO: 只有非picture的才会有title_tks字段
        batch_size = 32
        tts = [
            rmSpace(d["title_tks"]) 
            for d in docs if d.get("title_tks")
        ]
        cnts = [
            re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", 
                   " ", 
                   d["content_with_weight"]) 
            for d in docs
        ]
        tk_count = 0
        if len(tts) == len(cnts):
            tts_ = np.array([])
            for i in range(0, len(tts), batch_size):
                vts, c = emb_model.encode(None,tts[i: i + batch_size],{"batch_size":batch_size})
                if len(tts_) == 0:
                    tts_ = vts
                else:
                    tts_ = np.concatenate((tts_, vts), axis=0)
                tk_count += c
            tts = tts_

        cnts_ = np.array([])
        for i in range(0, len(cnts), batch_size):
            vts, c = emb_model.encode(cnts[i: i + batch_size])
            if len(cnts_) == 0:
                cnts_ = vts
            else:
                cnts_ = np.concatenate((cnts_, vts), axis=0)
            tk_count += c
        cnts = cnts_

        title_w = float(parser_config.get("filename_embd_weight", 0.1))
        vects = (title_w * tts + (1 - title_w) *
                cnts) if len(tts) == len(cnts) else cnts

        assert len(vects) == len(docs)
        for i, d in enumerate(docs):
            v = vects[i].tolist()
            d["q_%d_vec" % len(v)] = v
        return tk_count

    def chat(self,chat_model,system,message,**gen_conf):
        ans = chat_model.chat(system,message,gen_conf)
        return ans

