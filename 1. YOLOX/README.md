## Datasets
Save .json files under "./datasets/" 
  - mot17_val.json: https://drive.google.com/file/d/1JFwLgTQckApe-gvFYf1Je05DPKf5pNUx/view?usp=drive_link
  - mot17_test.json: https://drive.google.com/file/d/1EGDJ12QTDKbA1GChs7lNi9pmVXOBOktC/view?usp=drive_link
  - mot20_val.json: https://drive.google.com/file/d/12dwVn3qLO0L3IAOSJcVdYzpWhlOXmIFY/view?usp=drive_link
  - mot20_test.json: https://drive.google.com/file/d/1X6wyBOmtI6IFguAktNjoQ0zg0ZhNDGi2/view?usp=drive_link


## Model Zoo
Save weights files under "./weights/"
  - mot17_train_half.pth.tar: https://drive.google.com/file/d/1zQp2kLqCz25zMnOEIUSeS3i-mj_Wsj4Z/view?usp=drive_link
  - mot7_train.pth.tar: https://drive.google.com/file/d/1OBazmX6rLdOdgvbnhIcU2up9YL2O_B9T/view?usp=drive_link
  - mot20_train_half.pth.tar: https://drive.google.com/file/d/1QQ58QjcJUFmyxfamyDm2nvT06VmDT_1J/view?usp=drive_link
  - mot20_train.pth.tar: https://drive.google.com/file/d/16lDABxaV8SXibDDV-8Nq3O8NgYQ1Ahrb/view?usp=drive_link


## Run
Detection results will be created under "../outputs/1. det/" as pickle files
```
# For the MOT17 validation set
python detect.py -f "exps/yolox_x_mot17_val.py" -c "weights/mot17_train_half.pth.tar" -b 1 -d 1 -n "../outputs/1. det/MOT17_val_nms_0.8.pickle" --fp16 --fuse

# For the MOT17 test set
python detect.py -f "exps/yolox_x_mot17_test.py" -c "weights/mot7_train.pth.tar" -b 1 -d 1 -n "../outputs/1. det/MOT17_test_nms_0.8.pickle" --fp16 --fuse

# For the MOT20 validation
python detect.py -f "exps/yolox_x_mot20_val.py" -c "weights/mot20_train_half.pth.tar" -b 1 -d 1 -n "../outputs/1. det/MOT20_val_nms_0.8.pickle" --fp16 --fuse

# For the MOT20 test
python detect.py -f "exps/yolox_x_mot20_test.py" -c "weights/mot20_train.pth.tar" -b 1 -d 1 -n "../outputs/1. det/MOT20_test_nms_0.8.pickle" --fp16 --fuse
```

## Reference
This code is revised from https://github.com/Megvii-BaseDetection/YOLOX
