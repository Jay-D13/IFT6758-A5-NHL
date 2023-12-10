#!/bin/bash
IMAGE_NAME="serving:latest"

docker build -t $IMAGE_NAME -f Dockerfile.serving .