import os
from typing import *

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from omegaconf import OmegaConf

from text_classification.model import TransformerTextClassification

app = FastAPI()


class InferenceInput(BaseModel):
    text_list: List = ['جملات را در این لیست قرار دهید']


@app.on_event('startup')
def startup():
    global model

    config_path = os.getenv('config_path', './configs/sentiment.yaml')
    config = OmegaConf.load(config_path)
    model = TransformerTextClassification(config, mode='inference')


@app.post('/text_classification/inference')
def inference(inputs: InferenceInput):
    text_list = inputs.text_list
    try:
        outputs, exec_time = model.predict(text_list)
        response = {"outputs": outputs, "execution_time": exec_time}
    except Exception as e:
        raise HTTPException(status_code=504, detail=str(e))

    return response
