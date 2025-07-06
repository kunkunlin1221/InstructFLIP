devices=$1
python train.py \
	configs/rgb/ablations/loss/lambda3/instruct_flip_40_40_05_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda3/instruct_flip_40_40_10_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda3/instruct_flip_40_40_15_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda3/instruct_flip_40_40_20_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices