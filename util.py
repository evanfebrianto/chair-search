import googleapiclient.discovery
import time
import json
import numpy as np


def predict_json(project, model, input, version=None):
    """Send json data to a deployed model for prediction.
​
    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        input dict(dict()): Dictionary in the form of {'image_bytes': {"b64": input_string}}
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(name=name, body={'instances': input}).execute()
    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions'][0]['output_bytes']['b64']  # TODO: Make this just return json, not list


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken for {func.__name__} is {end - start} seconds")
        return result

    return wrapper

class PipelineTimer:
    def __init__(self):
        self.start = time.time()
        self.prev = self.start
        self.__data = {}
    
    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value):
        name, duration = value
        if name in self.__data:
            self.__data[name].append(duration)
        else:
            self.__data[name] = [duration]

    def __call__(self, name):
        now = time.time()
        duration = now - self.prev
        self.prev = now
        self.data = (name, duration)

    def get_statistic(self):
        # Calculate min, max, median, std, mean for each key
        result = {}
        for key, value in self.data.items():
            result[key] = {
                'min': np.min(value),
                'max': np.max(value),
                'median': np.median(value),
                'std': np.std(value),
                'mean': np.mean(value)
            }
        result['total_time'] = time.time() - self.start
        result['num of data'] = len(value)
        # remove 'start' key
        result.pop('start')
        return result
    
    def export(self, path):
        # export data from get_statistic to json file
        _temp = self.get_statistic()
        with open(path, 'w') as f:
            json.dump(_temp, f)
