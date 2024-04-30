
import os
import string
import secrets
import uuid


def get_project_path() -> str:

    path_info = os.path.abspath(__file__)
    for _ in range(4):
        path_info = os.path.dirname(path_info)
    return path_info


def gen_random_str(length=6):
    alphabet = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(alphabet) for _ in range(length))
    return random_string


def rmSpace(txt):
    txt = re.sub(r"([^a-z0-9.,]) +([^ ])", r"\1\2", txt, flags=re.IGNORECASE)
    return re.sub(r"([^ ]) +([^a-z0-9.,])", r"\1\2", txt, flags=re.IGNORECASE)

def gen_uuid():
    random_uuid = uuid.uuid4()
    return str(random_uuid)