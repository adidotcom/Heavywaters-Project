# Importing necessary modules
from sklearn.externals import joblib
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from flask import Flask
from flask import request
from flask import json

BUCKET_NAME = 'heavywatersdata'
MODEL_FILE_NAME = 'nbmodel.pkl'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
  payload = json.loads(request.get_data().decode('utf-8'))
  prediction = predict(payload['payload'])
  data = {}
  data['data'] = prediction[-1]
  return json.dumps(data)

def load_model():
  conn = S3Connection()
  bucket = conn.create_bucket(BUCKET_NAME)
  key_obj = Key(bucket)
  key_obj.key = MODEL_FILE_NAME

  contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
  return joblib.load(MODEL_LOCAL_PATH)

def predict(data):
  # Data to be processed here and sent for prediction
  return load_model().predict(data)
