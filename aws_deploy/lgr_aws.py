import pickle
import os
import argparse
import json

from sklearn.linear_model import LogisticRegression 
from model.test import A

def input_fn(input_data,content_type):
    
    if content_type=='application/json':
        data=json.loads(input_data)
        data=data['X']
        
        return data
    
def output_fn(output_data,content_type):
    
    if content_type=='application/json':
        pred={'predictions':output_data.tolist()}
        pred=json.dumps(pred)
        
        return pred
    
def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    
    a=A(2)
    print(a.a)
    
    file_path=os.path.join(model_dir, "model_lgr.pkl")
    model = pickle.load(open(file_path,'rb'))
    return model


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Logistic Regression Example')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args = parser.parse_args()
"""