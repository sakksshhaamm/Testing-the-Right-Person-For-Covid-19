from flask import Flask
app = Flask(__name__)
import pickle

file = open('model.pkl', 'wb')
clf = pickle.load(file)

@app.route('/')
def hello_world():

    #code for inference
    inputFeatures = [102, 1, 35, 1, 1, 1]
    infProb = clf.predict_proba([inputFeatures])[0][1]   
    return 'Hello, World!' + str(infProb)


    if __name__ == "__main__":
        app.run(debug=True)