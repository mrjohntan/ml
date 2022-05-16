from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd
import dill as pickle
import json
from utils.DataPreparation import *

app = FastAPI()


@app.post("/")
async def apicall(request: Request):
    try:
        test_json_dump = json.dumps(await request.json())
        test_df = pd.read_json(test_json_dump, orient='records')
        # Because of request processing Age is being considered as object, but it needs to be float type.
        # test_df['Age'] = test_df.Age.convert_objects(convert_numeric=True)
        test_df['Age'] = pd.to_numeric(test_df.Age)
        # Getting the PassengerId separated out
        passenger_ids = test_df['PassengerId']

    except Exception as e:
        print(':::: Exception occurred while reading json content ::::')
        raise e

    if test_df.empty:
        return("bad_request")
    else:
        # Load the saved model
        (f'Model loading..')
        loaded_model = None
        with open('./ml-model/voting_classifier_v1.pk', 'rb') as model:
            loaded_model = pickle.load(model)

        # Before we make any prediction, lets pre-process first.
        data_preparation = DataPreparation()
        test_df = data_preparation.preprocess(test_df)
        print(f'After pre-process test df - \n {test_df}')
        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(test_df)

        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame(
            {'PassengerId': passenger_ids, 'Survived': prediction_series})

        responses = JSONResponse(
            content=final_predictions.to_json(orient="records"))
        responses.status_code = 200

        return (responses)
