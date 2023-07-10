from .session_state import session
from .config import SAVE_FILEPATH
import pandas as pd

def store_data(result_db) -> None:
    """
    should be called after the result has been generated
    :return: None
    """
    df = pd.DataFrame(result_db)
    df.to_csv(SAVE_FILEPATH)
    
