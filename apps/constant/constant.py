
from enum import Enum

class Constant:

    image_save_prefix = "is-"
    


class CodeEnum(Enum):
    Success = 0
    Fail = 1

class LayoutType(Enum):
    general = "通用版面"
    text = "文本"
    table = "表格"