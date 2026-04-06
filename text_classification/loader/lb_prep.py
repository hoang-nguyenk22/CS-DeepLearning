from sklearn.preprocessing import MultiLabelBinarizer
import joblib

def get_mlb(src = "data/cs/mlb.pkl"):
    mlb=  joblib.load(src)
    return mlb, len(mlb.classes_)