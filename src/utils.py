from .session_state import session
from .config import SAVE_FILEPATH
import pandas as pd

def load_data() -> list:
    """
    load the result from previous users
    :return: list of dicts
    """
    df = pd.read_csv(SAVE_FILEPATH)
    result = df.to_dict('records', index=False)
    return result

def store_data(result_db) -> None:
    """
    should be called after the result has been generated
    :return: None
    """
    df = pd.DataFrame(result_db)
    df.to_csv(SAVE_FILEPATH)
    
