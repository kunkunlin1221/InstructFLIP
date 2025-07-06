devices=$1
python train.py \
	configs/rgb/ablations/loss/lambda2/instruct_flip_40_10_15_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda2/instruct_flip_40_20_15_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda2/instruct_flip_40_30_15_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices

python train.py \
	configs/rgb/ablations/loss/lambda2/instruct_flip_40_50_15_05.yaml \
	configs/rgb/protocols/orica/orica_mciowcs_instruct_b24.yaml \
	--devices=$devices