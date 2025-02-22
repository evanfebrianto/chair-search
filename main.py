import base64
import os, glob
from datetime import datetime
import random
import uuid
import json, time

from flask import Flask, render_template, request, redirect, jsonify
from google.cloud import storage, datastore

from vision.product_catalogue import get_similar_products, get_reference_image
from util import predict_json, PipelineTimer, preprocess_image, ProductCatalogue, Downloader
from config import config as cfg

if not os.getenv("RUNNING_ON_GCP"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.LOCAL_CREDENTIALS

app = Flask(__name__, static_folder='assets')

storage_client = storage.Client(project=cfg.PROJECT_ID)
datastore_client = datastore.Client(project=cfg.PROJECT_ID)

# Initialize variables
random_chairs = glob.glob("assets/static_chairs/*.json")
current_chair = None
pipeline = None
if cfg.DEBUG:
    pipeline = PipelineTimer()

downloader = Downloader(project_id=cfg.PROJECT_ID, bucket_name=cfg.BUCKET_NAME)

# Be careful with this line, it need to be re-run if you change the product images
# Create a ProductCatalogue object
catalogue = ProductCatalogue(project_id=cfg.PROJECT_ID, location_id=cfg.LOCATION_ID)
# Get the product mapping
product_mapping = catalogue.get_product_mapping()
downloader.download_blob_manager(list(product_mapping.values()))

@app.route("/index.html")
@app.route("/")
def root():
    return render_template("index.html")


@app.route('/autoDraw')
def auto_draw():
    if request.args.get('sketch_id'):
        if cfg.DEBUG:
            print('[DEBUG] Getting sketch from datastore with id: ', request.args.get('sketch_id'))
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

    # Make sure we don't get the same chair
    global current_chair
    chair = random.choice(random_chairs)
    while chair == current_chair:
        chair = random.choice(random_chairs)
    current_chair = chair
    resp = {
        "success": True,
        "filepath": current_chair
    }
    print(f'[INFO] Sending chair: {current_chair}')
    return jsonify(resp)


@app.route("/generate", methods=["POST"])
def generate():
    global pipeline, product_mapping, downloader
    if cfg.DEBUG:
        max_iter = cfg.STRESS_TEST_ITERATIONS
        pipeline = PipelineTimer()
    else:
        max_iter = 1
    for i in range(max_iter):
        if pipeline:
            print(f'[INFO] Iteration {i+1}/{max_iter}')
            pipeline('start')

        sketch = request.json["imgBase64"]
        if pipeline:
            pipeline('downloaded sketch')

        cropped_sketch = json.loads(preprocess_image(sketch.split(',')[1]))
        if pipeline:
            pipeline('preprocessed sketch')

        generated_chair = predict_json(project="chair-search-demo", model="chair_generation",
                                    input=cropped_sketch, version=cfg.MODEL_VERSION)
        if pipeline:
            pipeline('GAN generated chair')

        # Get similar products and filter to top 3
        similar_products, response = get_similar_products(cfg.PRODUCT_SET_ID, generated_chair)  # Generated chair
        if pipeline:
            pipeline('got similar products')

        top = sorted(similar_products, key=lambda product: product.score, reverse=True)[:4]
        if pipeline:
            pipeline('filtered similar products')

        products = [(product.product.display_name, product.image, product.score) for product in
                    top] or "No matching products found!"
        # print(f'[DEBUG] Products: {products}')
        if pipeline:
            pipeline('got product info')

        images = []
        for (product_name, product_image, _) in products:
            image_source = product_mapping[product_image]
            images.append({'name': product_name, 'src': downloader.blob_images[image_source]})
        if pipeline:
            pipeline('got product images')
            # if current_chair:
            #     pipeline.export(path=f"improvement_results/{current_chair.split('/')[-1].split('.')[0]}.json")
            #     upload_json(storage_client, cfg.TEST_RESULT_BUCKET, f"{current_chair.split('/')[-1].split('.')[0]}.json", json.dumps(pipeline.get_statistic()))

        generated_chair = add_png_header(generated_chair)
        resp = {
            "success": True,
            "results": images,
            "original_sketch": sketch,
            "generated_chair": generated_chair
        }
    return jsonify(resp)


def download_blob(client, bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    b64_img = blob.download_as_string()
    return b64_img

def upload_json(client, bucket_name, destination_blob_name, data):
    """Uploads a file to the bucket."""
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(data, content_type='application/json')

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
    app.run(host='127.0.0.1', port=8080, debug=True, use_reloader=False)
    # Add labels to chairs and use "Jag vill ha dyna" "jag vill ha armstöd" as labels and use these for product catalogue
