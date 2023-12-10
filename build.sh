#!/bin/bash
IMAGE_NAME="ift6758/serving:latest"

docker build -t $IMAGE_NAME -f Dockerfile.serving .