docker run \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --net=host \
    -v $PWD:$PWD \
    -v $HOME/.cache:/root/.cache \
    -v /data:/data \
    -it --rm instructflip $@
# monut data folder by yourself