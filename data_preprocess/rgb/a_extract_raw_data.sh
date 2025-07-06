docker run \
    --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --net=host \
    -v /data:/data\
    -v $PWD:/code \
    -it --rm InstructFLIP_data ipython --pdb -- a_extract_raw_data.py $@