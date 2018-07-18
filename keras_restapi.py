#!/usr/bin/env python2
from keras.applications import ResNet50, VGG19
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import flask
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
server = flask.Flask(__name__)

def run_model():
	logger.info("Preparing CNN and loading pretrainned dataset")
	global model
	model = VGG19(weights="imagenet")
	logger.info("Model loaded and dataset ready")
	global graph
	graph = tf.get_default_graph()
	logger.info("Model ready for use")

def process_image(img=False, target=False):
	logger.info("Preparing image for the clasificator")
	if not img or not target:
		return False

	logger.info("Checking image mode and changing to RGB")
	if img.mode not in ["RGB"]:
		img = img.convert("RGB")

	logger.info("Adjusting image size")
	img = img.resize(target)
	logger.info("Converting img in array")
	img = img_to_array(img)
	logger.info("Numpy is now expanding the axis of the array")
	img = np.expand_dims(img, axis=0)
	logger.info("Preprocess image for the input")
	img = imagenet_utils.preprocess_input(img)
	return img

@server.route("/rank", methods=["POST"])
def predict():
	result = {"success": False}

	if flask.request.method in ["POST"]:
		if flask.request.files.get("img"):
			img = flask.request.files["img"].read()
			img = Image.open(io.BytesIO(img))
			img = process_image(img, target=(224, 224))

			with graph.as_default():
				preds = model.predict(img)
				results = imagenet_utils.decode_predictions(preds)
				result["predictions"] = []

				for (imagenetID, label, prob) in results[0]:
					r = {"label": label, "probability": float(prob)}
					result["predictions"].append(r)

				result["success"] = True

	return flask.jsonify(result)


if __name__ == "__main__":
	logger.info("Loading resources and starting server")
	run_model()
	server.run()
