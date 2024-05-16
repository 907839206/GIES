
from typing import List

def calc_avg(data:List[float]) -> float:
    if isinstance(data, dict):
        data = data["data"]
    if len(data) == 0:
        return 0
    return sum(data) / len(data)
    
def calc_max(data:List[float]) -> float:
    if isinstance(data, dict):
        data = data["data"]
    return max(data)

def calc_min(data:List[float]) -> float:
    if isinstance(data, dict):
        data = data["data"]
    return min(data)