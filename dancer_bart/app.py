from flask import Flask, request
from flask_json import FlaskJSON, json_response

app = Flask(__name__)
json = FlaskJSON()
app.config['JSON_ADD_STATUS'] = False


@app.route('/')
def display():
   return 'I am displaying'

@app.route('/upload-file', methods=['POST'])
def upload_file():
   return 'File uploading'

@app.route('/upload-text', methods=['POST'])
def upload_text():
   text = request.json['content']
   summary = generate_summary(text)
   return json_response(status_=200, summary=summary)


def generate_summary(text):
   return "This is my summary"


# Init model
import torch
import utils
