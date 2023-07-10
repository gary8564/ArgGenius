import streamlit as st


class Session:
    def __init__(self) -> None:
        super().__init__()

    def init(self, key: str, value):
        if key not in st.session_state:
            st.session_state[key] = value

    def update(self, key: str, value):
        st.session_state[key] = value

    def has(self, key: str):
        return key in self.to_dict().keys()

    def get(self, key: str):
        tmp = self.to_dict()
        if key in tmp.keys():
            return tmp[key]
        else:
            return None

    def summary(self):
        tmp = self.to_dict()
        num = len(tmp.keys())
        ret = f'''There are {num} variables:\n\n'''
        for k in tmp.keys():
            v = tmp[k]
            ret += f'''"{k}": {v}\n\n'''
        return ret

    def to_dict(self):
        return st.session_state.to_dict()

    def render(self):
        page = self.get_current_page()
        page_func = self.get_page_map()[page]
        page_func()

    def clear(self):
        session.update("arguAI", '')
        session.update("arguHuman", '')
        session.update("status", 0)
        session.update("winner", '')
    
    def remove(self, key: str):
        del st.session_state[key]


session = Session()