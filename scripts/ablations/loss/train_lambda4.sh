devices=$1
python train.py \
	configs/rgb/ablations/loss/lambda4/instruct_flip_40_40_15_01.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda4/instruct_flip_40_40_15_03.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda4/instruct_flip_40_40_15_07.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda4/instruct_flip_40_40_15_09.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices