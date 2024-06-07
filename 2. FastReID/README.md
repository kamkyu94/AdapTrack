## Model Zoo
Save weights files under "./weights/"
  - mot17_sbs_S50.pth: https://drive.google.com/file/d/1XpC27lWBL-wSf-9ceh2fsnAQeOGlirig/view?usp=drive_link
  - mot20_sbs_S50.pth: https://drive.google.com/file/d/1UiVMWtGf-ktGRUFRfp2L5UaAiUk8jZCR/view?usp=drive_link


## Run
Detection + feature extractionResults will be created under "../outputs/2. det_feat/" as pickle files
```
# For MOT17 validation
python ext_feats.py --dataset "mot17" --pickle_path "../outputs/1. det/MOT17_val.pickle" --output_path "../outputs/2. det_feat/MOT17_val.pickle" --data_path "../../dataset/MOT17/train/"

# For MOT17 test
python ext_feats.py --dataset "mot17" --pickle_path "../outputs/1. det/MOT17_test.pickle" --output_path "../outputs/2. det_feat/MOT17_test.pickle" --data_path "../../dataset/MOT17/test/"

# For MOT20 validation
python ext_feats.py --dataset "mot20" --pickle_path "../outputs/1. det/MOT20_val.pickle" --output_path "../outputs/2. det_feat/MOT20_val.pickle" --data_path "../../dataset/MOT20/train/"

# For MOT20 test
python ext_feats.py --dataset "mot20" --pickle_path "../outputs/1. det/MOT20_test.pickle" --output_path "../outputs/2. det_feat/MOT20_test.pickle" --data_path "../../dataset/MOT20/test/"
```

## Reference
This code is revised from https://github.com/JDAI-CV/fast-reid
