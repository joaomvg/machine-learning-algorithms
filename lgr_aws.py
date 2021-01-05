import pickle
import os
import argparse

from sklearn.linear_model import LogisticRegression

def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    file_path=os.path.join(model_dir, "model_lgr.pkl")
    model = pickle.load(open(file_path,'rb'))
    return model

def input_fn():

def output_fn(prediction, content_type)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Logistic Regression Example')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args = parser.parse_args()
    