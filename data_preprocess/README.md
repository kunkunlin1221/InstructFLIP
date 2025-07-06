# Data preparation

## build docker

```bash
cd docker
bash build.sh
cd ..
```

## For CA dataset

```bash
bash ca/make_aligh_faces.sh $data_folder $output_folder
```

## For MCIOWCS

```bash
cd rgb
bash a_extract_raw_data.sh $data_txt_dir $raw_data_dir $dst_dir
```
