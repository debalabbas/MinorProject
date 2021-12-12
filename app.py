
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
# from keras.models import load_model

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model-checkpoint/roberta_trained/'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def get_pred(sent1,sent2):
        
    tokens = tokenizer(sent1,sent2,return_tensors='pt')
    
    output = model(**tokens)
    
    index = np.argmax(output.logits.detach().numpy(),axis=1)[0]

    if index == 0:
        return sent1
    else:
        return sent2


# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the data from post request
        data = request.form.to_dict()
        print(data)
        sent0 = data['sent1']
        sent1 = data['sent2']
        print('--------------------------------------------------------')
        print(sent0,sent1)
        print('--------------------------------------------------------')
        # Make prediction
        corr = get_pred(sent0, sent1)
                
        return render_template('index.html', result = {'Correct Sentence': corr})

    return None


if __name__ == '__main__':
    app.run()