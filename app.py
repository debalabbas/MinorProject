
from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import load_model

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = ''

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(sent0, sent1, model):
    
    x = tokenizer(sent0, sent1)
    pred_label = model.predict(x)
    if pred_label == 0:
        return sent0
    else:
        return sent1
    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the data from post request
        sent0 = request.get['Sentence 0']
        sent1 = request.get['Sentence 1']
        
        # Make prediction
        corr = model_predict(sent0, sent1, model)
                
        return render_template('index.html', result = {'Correct Sentence': corr})

    return None


if __name__ == '__main__':
    app.run(debug=True)