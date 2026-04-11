from dataclasses import dataclass

class Emb_conf:
    def __init__(self, ds="eurlex"):
        self.dim = 384
        self.model_name=f'sentence-transformers/all-MiniLM-L{"12" if ds == "eurlex" else "6"}-v2'


emb_cs = Emb_conf("cs")
emb_eur = Emb_conf("eurlex")