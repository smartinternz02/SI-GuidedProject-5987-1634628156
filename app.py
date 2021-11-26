from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.models import Sequential
#import required libraries
import numpy as np
import os

#import Flask
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#import keras
#import keras
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

#graph = tf.get_default_graph()


app = Flask(__name__)
model = load_model("wild.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        #with graph.as_default():
        #preds = model.predict_classes(x)
        preds =np.argmax(model.predict(x), axis=1)
        print("prediction",preds)
            
        index=['Mountain Laurel_nonedible', 'Peppergrass_edible', 'Purple Deadnettle_edible', 
       'Rhododendron_nonedible', 'Toothwort_edible', 'Wild Grape Vine_edible', 'Wild Leek_edible', 'rattlebox_nonedible']
        text = "prediction : "+ index[preds[0]]
        return text
    
if __name__ == '__main__':
    app.run(debug = False)
        
        
        
    
    
    