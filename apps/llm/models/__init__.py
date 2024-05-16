

from .qwen import Qwen
from .openai import Openai

LLMDrawer = {
    "QWen": Qwen,
    "Openai":Openai
}