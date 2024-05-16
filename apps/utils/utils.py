
import uuid, hashlib

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def generate_uuid():
    return str(uuid.uuid4())

def calculate_md5(info):
    md5_obj = hashlib.md5()
    md5_obj.update(info.encode('utf-8'))
    return md5_obj.hexdigest()