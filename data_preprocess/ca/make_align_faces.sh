docker run --gpus all \
	-v $PWD:/code \
	-v /data:/data \
	-it InstructFLIP_data ipython --pdb -- data_preprocess/ca/make_align_faces.py $@