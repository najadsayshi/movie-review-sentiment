from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib

app = FastAPI()


@app.get("/")
def read_index():
    return FileResponse("index.html")
# TODO: load model.pkl and vectorizer.pkl here, once, at startup
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# a request body model — defines what JSON shape the API expects
class ReviewRequest(BaseModel):
    review : str

@app.post('/predict')
def predict(request : ReviewRequest):
    text = request.review
    number = vectorizer.transform([text])
    prediction = model.predict(number)
    confidence = model.predict_proba(number)
    if prediction[0]==0:
        return {'answer' : 'negative',
                'confidence' : confidence[0][0]}
    else:
        return {'answer' : 'positive',
                'confidence' : confidence[0][1]} 