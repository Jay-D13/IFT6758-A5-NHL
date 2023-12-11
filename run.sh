#!/bin/bash
IMAGE_NAME="ift6758/serving:latest"

COMET_API_KEY=${COMET_API_KEY}
FLASK_LOG=${FLASK_LOG:-"flask.log"}

docker run -e COMET_API_KEY=$COMET_API_KEY -e FLASK_LOG=$FLASK_LOG -p 5000:5000 $IMAGE_NAME