import base64
import os
from datetime import datetime
import random
from urllib import response
import uuid
import json
import time

from flask import Flask, render_template, request, redirect, jsonify
from google.cloud import storage, datastore
import requests

from vision.product_catalogue import get_similar_products, get_reference_image, sample_get_reference_image
from util import predict_json, time_it
from config import config as cfg

if not os.getenv("RUNNING_ON_GCP"):
    print("Running locally")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.LOCAL_CREDENTIALS
else:
    print("Running on GCP")

app = Flask(__name__, static_folder='assets')

storage_client = storage.Client(project=cfg.PROJECT_ID)
datastore_client = datastore.Client(project=cfg.PROJECT_ID)


@app.route("/index.html")
@app.route("/")
def root():
    return render_template("index.html")


@app.route('/autoDraw')
@time_it
def auto_draw():
    if request.args.get('sketch_id'):
        source_blob_name = f"{request.args.get('sketch_id')}/coordinates.json"
        sketch_coords = json.loads(download_blob(storage_client, cfg.SKETCH_BUCKET, source_blob_name))

        # Backwards compatibility check
        if isinstance(sketch_coords['drag'], str):
            for key, value in sketch_coords.items():
                sketch_coords[key] = value.split(',')
            sketch_coords['drag'] = ['true' in x for x in sketch_coords['drag']]
        
        resp = {
            "success": True,
            "x": sketch_coords['x'],
            "y": sketch_coords['y'],
            "drag": sketch_coords['drag']
        }
        return jsonify(resp)

    files = os.listdir("assets/static_chairs")
    file = files[random.randint(0, len(files) - 1)]
    path = f"assets/static_chairs/{file}"
    resp = {
        "success": True,
        "filepath": path
    }
    return jsonify(resp)


@app.route("/generate", methods=["POST"])
def generate():
    print(f'[DEBUG] Generate request is called!')
    T1 = time.time()
    sketch = request.json["imgBase64"]
    T2 = time.time()
    # This code below saves the drawn chair locally
    # with open("huijie_something_3.json", 'w') as file:
    #     import json
    #     # request.json.pop('imgBase64')
    #     local_img = {
    #         'x': request.json['x'],
    #         'y': request.json['y'],
    #         'drag': request.json['drag']
    #     }

    #     local_img['x'] = [str(number) for number in local_img['x']]
    #     local_img['y'] = [str(number) for number in local_img['y']]
    #     file.write(json.dumps(local_img))

    T3 = time.time()
    payload = {
        "img": sketch.split(',')[1]
    }
    cropped_sketch = requests.post(cfg.PREPROCESS_URL, json=payload).json()
    T4 = time.time()
    generated_chair = predict_json(project="chair-search-demo", model="chair_generation",
                                   input=cropped_sketch, version=cfg.MODEL_VERSION)
    T5 = time.time()

    # Get similar products and filter to top 3
    similar_products, response = get_similar_products(cfg.PRODUCT_SET_ID, generated_chair)  # Generated chair
    T6 = time.time()

    # For debugging purposes
    # export similar_products which contains list of dicts
    # with open("similar_products.json", 'w') as file:
    #     from google.protobuf.json_format import MessageToJson, MessageToDict
    #     serialized = MessageToJson(response._pb)
    #     _dict = MessageToDict(response._pb)
    #     # file write _dict
    #     file.write(json.dumps(_dict))

    T7 = time.time()
    top = sorted(similar_products, key=lambda product: product.score, reverse=True)[:4]
    products = [(product.product.display_name, product.image, product.score) for product in
                top] or "No matching products found!"
    T8 = time.time()
    images = []
    for index, (product_name, product_image, product_score) in enumerate(products):
        img_uri = get_reference_image(product_image).uri.split('/')
        blob_name = os.path.join(*img_uri[3:])
        img_blob = download_blob(storage_client, cfg.BUCKET_NAME, blob_name)
        img_blob = base64.b64encode(img_blob).decode()  # Convert to string so we can add data URI header
        img_blob = add_png_header(img_blob)
        images.append({'name': product_name, 'src': img_blob})
    T9 = time.time()

    generated_chair = add_png_header(generated_chair)
    resp = {
        "success": True,
        "results": images,
        "original_sketch": sketch,
        "generated_chair": generated_chair
    }
    T10 = time.time()
    print(f'[DEBUG] T2-T1: {T2-T1}')
    print(f'[DEBUG] T3-T2: {T3-T2}')
    print(f'[DEBUG] T4-T3: {T4-T3}')
    print(f'[DEBUG] T5-T4: {T5-T4}')
    print(f'[DEBUG] T6-T5: {T6-T5}')
    print(f'[DEBUG] T7-T6: {T7-T6}')
    print(f'[DEBUG] T8-T7: {T8-T7}')
    print(f'[DEBUG] T9-T8: {T9-T8}')
    print(f'[DEBUG] T10-T9: {T10-T9}')
    print(f'[DEBUG] Total time taken: {T10-T1}')
    return jsonify(resp)


def download_blob(client, bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    b64_img = blob.download_as_string()
    return b64_img


def upload_blob(client, bucket_name, blob, destination_blob_name):
    """Uploads blob to gcs"""
    # Append image and a timestamp
    destination_blob_name = os.path.join(destination_blob_name, f"sketch_{datetime.utcnow()}")
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(blob)


def add_png_header(data):
    return "data:image/png;base64," + data


@app.route("/send-sketch", methods=["POST"])
def send_sketch():
    """Save sketch and sketch coordinates to Datastore and GCS"""
    # TODO: Make this prettier :)
    print(f'[DEBUG] Send sketch request is called!')
    req = request.get_json()
    email = req.get('email')
    name = req.get('name')
    sketch = req.get('sketch')
    coords = {
        'x': req.get('x').split(','),
        'y': req.get('y').split(','),
        'drag': ["true" in x for x in req.get('drag').split(',')]
    }

    coords = json.dumps(coords)
    if not email or not name:
        return redirect("index.html")
    kind = 'DemoUser'
    ds_id = str(uuid.uuid4())
    upload_image(ds_id, sketch, coords)
    demo_user_key = datastore_client.key(kind, ds_id)
    demo_user = datastore.Entity(key=demo_user_key)
    demo_user['name'] = name
    demo_user['email'] = email
    demo_user['gcs_link'] = f"gs://{cfg.SKETCH_BUCKET}/{ds_id}/"

    datastore_client.put(demo_user)

    return redirect("index.html")


def upload_image(ds_id, img, coords):
    img_dest_blob = f"{ds_id}/base64_image.txt"
    coords_dest_blob = f"{ds_id}/coordinates.json"
    bucket = storage_client.get_bucket(cfg.SKETCH_BUCKET)
    img_blob = bucket.blob(img_dest_blob)
    coords_blob = bucket.blob(coords_dest_blob)

    img_blob.upload_from_string(img)
    coords_blob.upload_from_string(coords)


if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host="127.0.0.1", port=8080, debug=True)
    # Add labels to chairs and use "Jag vill ha dyna" "jag vill ha armstöd" as labels and use these for product catalogue
