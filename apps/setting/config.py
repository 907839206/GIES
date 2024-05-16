import yaml,os,logging,traceback

from utils import is_colab

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

abspath = os.path.abspath(__file__)

def load_config():
    if is_colab():
        env = "stage"
    else:
        env = os.environ.get("ENV","stage")
    full_config_path = os.path.join(
        os.path.dirname(abspath),
        f"config.{env}.yaml"
    )
    if not os.path.isfile(full_config_path):
        raise ValueError(f"{full_config_path} must be a config file!")
    
    with open(full_config_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
            return data
        except yaml.YAMLError as _:
            logger.error(f"read config file error, msg:{traceback.format_exc()}")
            return None


def init_config(obj, info):
    for key, value in info.items():
        if isinstance(value, dict):  
            nested_class = type(key.capitalize(), (object,), value)
            setattr(obj, key, nested_class())
            init_config(getattr(obj, key), value)
        else:
            setattr(obj, key, value)
    return obj

config_dict = load_config()
dyn_class = type('Config', (object,), config_dict)
obj = dyn_class()
config = init_config(obj,config_dict)
