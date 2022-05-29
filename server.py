import argparse

import uvicorn

from typing import *

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from omegaconf import OmegaConf

from predict import Predictor

app = FastAPI()

opposition_config = OmegaConf.load('configs/security.yaml')
opposition_model = Predictor(opposition_config)

security_config = OmegaConf.load('configs/security.yaml')
security_model = Predictor(security_config)

models = {'security': security_model, 'opposition': opposition_model}
configs = {'security': security_config, 'opposition': opposition_config}


class JsonInput(BaseModel):
    task: str = Literal['security', 'opposition']
    include_meta: bool = True
    output_type: str = 'prob_all_classes'
    output_format: str = 'dict'
    data: list = [{"text": "جمله اول را اینجا وارد کنید"},
                  {"text": "جمله دوم را اینجا وارد کنید و با همین فرمت جملات بیشتر را وارد کنید"}]


@app.post('/shahab/opposition')
def opposition(inputs: JsonInput):
    return shahab_classification(inputs, task='opposition')


@app.post('/shahab/security')
def opposition(inputs: JsonInput):
    return shahab_classification(inputs, task='security')


def shahab_classification(inputs, task):
    model = models[task]
    idx2label = model.idx2label
    text_list = inputs.data

    results = {'result': [],
               'errors': {},
               'meta': {'response_length': 0,
                        'elapsed_time(s)': 0.,
                        'speed (text/s)': 0.},
               'status': 504}

    total_exec_time = 0.
    for text_dict in text_list:
        text = text_dict['text']
        logits, exec_time = model(text)
        class_probs_list = create_output_probs_dict(logits, idx2label)
        result = {'response': {'predicted_list': [], 'text': text}}
        for class_prob_dict in class_probs_list:
            result['response']['predicted_list'].append({'predicted_class': class_prob_dict['predicted_class'],
                                                         'probability': class_prob_dict['probability']})
        total_exec_time += exec_time

        results['result'].append(result)
        results['meta']['response_length'] = len(text_list)
        results['meta']['elapsed_time(s)'] = total_exec_time
        results['meta']['speed (text/s)'] = len(text_list) / total_exec_time
        results['status'] = 200

    return results


def create_output_probs_dict(logits: np.array, idx2label_mapper):
    sorted_indices = np.argsort(-logits)  # sort by descending order
    logits = logits[sorted_indices]
    logit_indice_stack = np.column_stack((sorted_indices, logits))
    output_list = []
    for index, prob in logit_indice_stack:
        label = idx2label_mapper[index]
        d = {'predicted_class': label,
             'probability': prob}
        output_list.append(d)
    return output_list
