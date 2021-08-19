# Handy commands:
# - `make docker-build`: builds DOCKERIMAGE
PROJECT ?= permatrack
WORKSPACE ?= /workspace/$(PROJECT)
DOCKER_IMAGE ?= ${PROJECT}:latest

SHMSIZE ?= 444G
DOCKER_OPTS := \
			--name ${PROJECT} \
			--rm -it \
			--shm-size=${SHMSIZE} \
			-e AWS_DEFAULT_REGION \
			-e AWS_ACCESS_KEY_ID \
			-e AWS_SECRET_ACCESS_KEY \
			-e HOST_HOSTNAME= \
			-e NCCL_DEBUG=VERSION \
            -e DISPLAY=${DISPLAY} \
            -e XAUTHORITY \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
			-v ~/.aws:/root/.aws \
			-v /root/.ssh:/root/.ssh \
			-v ~/.cache:/root/.cache \
			-v /data:/data \
			-v /mnt/fsx/:/mnt/fsx \
			-v /dev/null:/dev/raw1394 \
			-v /tmp:/tmp \
			-v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v ${PWD}:${WORKSPACE} \
			-w ${WORKSPACE} \
			--privileged \
			--ipc=host \
			--network=host

NGPUS=$(shell nvidia-smi -L | wc -l)


.PHONY: all clean docker-build

all: clean

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

docker-build:
	docker build \
		-f docker/Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-start-interactive: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} bash

docker-run: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "${COMMAND}"