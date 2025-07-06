devices=$1
python train.py \
	configs/rgb/ablations/loss/lambda1/instruct_flip_10_40_15_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda1/instruct_flip_20_40_15_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda1/instruct_flip_30_40_15_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda1/instruct_flip_50_40_15_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices