import pickle

def save_obj(obj, src: str):
    with open(src, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(src: str):
    with open(src, 'rb') as dic_f:
        obj = pickle.load(dic_f)
    return obj
