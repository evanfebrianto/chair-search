from google.cloud import vision_v1p3beta1
from google.cloud import storage
from multiprocessing import Pool, Process, Queue
import googleapiclient.discovery
from skimage.morphology import thin
import time
import json
import numpy as np
import cv2
import base64
import os

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
    """
    A wrapper function to time a function and can be used as a custom decorator
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken for {func.__name__} is {end - start} seconds")
        return result

    return wrapper

def crop_and_resize(src):
    """
        crop edge image to discard white pad, and resize to training size
        based on: https://stackoverflow.com/questions/48395434/how-to-crop-or-remove-white-background-from-an-image
        [OBS!] only works on image with white background
    """
    height, width, _ = src.shape

    # (1) Convert to gray, and threshold
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    # (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    # (4) Crop
    x, y, w, h = cv2.boundingRect(cnt)
    x_1 = max(x, x - 10)
    y_1 = max(y, y - 10)
    x_2 = min(x + w, width)
    y_2 = min(y + h, height)
    dst = gray[y_1:y_2, x_1:x_2]
    # pad white to resize
    height = int(max(0, w - h) / 2.0)
    width = int(max(0, h - w) / 2.0)
    padded = cv2.copyMakeBorder(dst, height, height, width, width, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return cv2.resize(padded, (256, 256), interpolation=cv2.INTER_NEAREST)

def preprocess_image(base64_string):
    img_bytes=base64.b64decode(base64_string)
    src = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
    
    # Crop the sketch and minimize white padding.
    cropped = crop_and_resize(src)
    # Skeletonize the lines
    skeleton = thin(cv2.bitwise_not(cropped))
    final = np.asarray(1 - np.float32(skeleton))
    fixed_channel = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    
    _, img_png = cv2.imencode('.png', fixed_channel * 255)
    encoded_input_string = base64.b64encode(img_png.tobytes())
    return json.dumps({'image_bytes': {"b64": encoded_input_string.decode()}})


class PipelineTimer:
    """
    PipelineTimer is a class to time the pipeline
    """
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


class ProductCatalogue:
    def __init__(self, project_id, location_id):
        self.project_id = project_id
        self.location_id = location_id

        self.client_vision = vision_v1p3beta1.ProductSearchClient()
        self.products = self.sample_list_products()

        self.product_mapping = {}
        self.__queue = Queue()

    def sample_list_products(self):
        # print(f'[INFO] Running sample_list_products')

        # Initialize request argument(s)
        request = vision_v1p3beta1.ListProductsRequest(
            parent=f'projects/{self.project_id}/locations/{self.location_id}',
        )

        # Make the request
        page_result = self.client_vision.list_products(request=request)

        # Handle the response
        result = []
        for response in page_result:
            result.append(response.name)

        return result


    def sample_list_reference_images(self, parent):
        # print(f'[INFO] Running sample_list_reference_images')

        # Initialize request argument(s)
        request = vision_v1p3beta1.ListReferenceImagesRequest(
            parent=parent,
        )

        # Make the request
        page_result = self.client_vision.list_reference_images(request=request)

        # Handle the response
        for response in page_result:
            _resp = {response.name: os.path.join(*response.uri.split('/')[3:])}
            self.__queue.put(_resp)

    def get_product_mapping(self):
        processes = []

        for product in self.products:
            p = Process(target=self.sample_list_reference_images, args=(product,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        while not self.__queue.empty():
            self.product_mapping.update(self.__queue.get())

        return self.product_mapping

class Downloader:
    def __init__(self, project_id, bucket_name):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.blob_images = {}
        self.__queue = Queue()

    @staticmethod
    def download_blob(project_id, bucket_name, image_source):
        print(f'[INFO] Running download_blobs')
        client_bucket = storage.Client(project=project_id)
        bucket = client_bucket.get_bucket(bucket_name)
        blob = bucket.blob(image_source)
        blob = blob.download_as_string()
        blob = "data:image/png;base64," + base64.b64encode(blob).decode()
        return {image_source: blob}

    def get_blob_images(self, images_to_download: list):
        # run download_blob in parallel
        with Pool(processes=4) as pool:
            results = pool.starmap(self.download_blob, [(self.project_id, self.bucket_name, image_source) for image_source in images_to_download])
        return {k: v for d in results for k, v in d.items()}

    def download_blob_parallel(self, image_source):
        self.__queue.put(self.download_blob(self.project_id, self.bucket_name, image_source))

    def get_blob_images_parallel(self, images_to_download: list):
        processes = []

        for image_source in images_to_download:
            p = Process(target=self.download_blob_parallel, args=(image_source,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        while not self.__queue.empty():
            self.blob_images.update(self.__queue.get())

        return self.blob_images