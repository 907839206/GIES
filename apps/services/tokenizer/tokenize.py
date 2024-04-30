
import re

def tokenize(d, t):
    d["content_with_weight"] = t
    # TODO：特征提取
    # t = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", t)
    # d["content_ltks"] = huqie.qie(t)
    # d["content_sm_ltks"] = huqie.qieqie(d["content_ltks"])