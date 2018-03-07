# ------------------------- Importing module ----------------------------
from flask import Flask, request, json
import boto3
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# --------------- Creating a Flask object and connecting to S3 -----------
BUCKET_NAME = 'newheavywaters'
MODEL_FILE_NAME = 'nbmodel.pkl' 
app = Flask(__name__) 
S3 = boto3.client('s3', region_name='us-east-2')  
@app.route('/', methods=['POST'])
def index():    
    # Parse request body for model input 
    body_dict = request.get_json()    
    data = body_dict['data'] 
    
    final_data = tfd_model(data)
    # Load model
    model = load_model(MODEL_FILE_NAME)   
    # Make prediction 
    prediction = model.predict(final_data).tolist()  
    predictions = pd.Series(prediction).to_json(orient='values')
    print prediction
    # Respond with prediction result
    result = {'prediction': predictions}    
   
    return json.dumps(result)

def load_model(key):    
    # Load model from S3 bucket
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)  
    # Load pickle model
    model_str = response['Body'].read()
    
    model = pickle.loads(model_str)     
    return model

def tfd_model(data):
    vec = joblib.load('vec_count.joblib')
    X = vec.transform([data])
    return X
        
if __name__ == '__main__':    
    # listen on all IPs 
    app.run(host='0.0.0.0')
    
    
    
