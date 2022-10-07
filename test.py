from util import PipelineTimer
import time
import json

pipeline = PipelineTimer()
def random_func():
    pipeline('start')
    print('Running pipeline 1')
    time.sleep(2)
    pipeline('pipeline 1')
    print('Running pipeline 2')
    time.sleep(2)
    pipeline('pipeline 2')
    print('Running pipeline 3')
    time.sleep(2)
    pipeline('pipeline 3')

if __name__ == '__main__':
    for i in range(3):
        random_func()
    # print(pipeline.data)
    print(json.dumps(pipeline.get_statistic()))
    pipeline.export('test.json')