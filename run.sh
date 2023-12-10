#!/bin/bash

IMAGE_NAME="votre_nom_dimage"

COMET_API_KEY=${COMET_API_KEY}
FLASK_LOG=${FLASK_LOG:-"flask.log"}

docker run -e COMET_API_KEY=$COMET_API_KEY -e FLASK_LOG=$FLASK_LOG -p 5000:5000 $IMAGE_NAME
