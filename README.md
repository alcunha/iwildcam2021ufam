# iWildcam 2021 - UFAM Team

### Requirements

Prepare an environment with python=3.8, tensorflow=2.3.1

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

### Data

Please refer to the [iWildCam 2021 Github page](https://github.com/visipedia/iwildcam_comp) for additional dataset details and download links.

### Training

To train a classifier, use the script `classification/train.py`. Since we use multiple training stages, the script `classification/multi_stage_train.py` can be used to automate that:
```bash
python multi_stage_train.py --annotations_json=PATH_TO_BE_CONFIGURED/iwildcam2021_train_annotations.json \
    --megadetector_results_json=PATH_TO_BE_CONFIGURED/iwildcam2021train_originalimage_megadetector_v4.1_results_parsed.json \
    --dataset_dir=PATH_TO_BE_CONFIGURED/ \
    --model_name=efficientnet-b2 \
    --input_size=260 \
    --input_size_stage3=380 \
    --input_scale_mode=uint8 \
    --batch_size=32 \
    --lr_stage1=0.01 \
    --lr_stage2=0.01 \
    --lr_stage3=0.001 \
    --momentum=0.9 \
    --epochs_stage1=4 \
    --epochs_stage2=20 \
    --epochs_stage3=2 \
    --unfreeze_layers=18 \
    --label_smoothing=0.1 \
    --randaug_num_layers=6 \
    --randaug_magnitude=2 \
    --random_seed=42 \
    --model_dir=PATH_TO_BE_CONFIGURED/
```

The parameters can also be passed using a config file:
```bash
python multi_stage_train.py --flagfile=configs/efficientnet_b2_multicrop_bags_final_submission_training.config
```

#### Training using full image

By default, the training script trains models using bounding boxes crops from MegaDetector predictions. To use the full image, please specify --use_full_image.

#### Training using Balanced Group Softmax

To train a model using [Balanced Group Softmax](https://arxiv.org/abs/2006.10408):

1. Train a model, as usual, using `classification/train.py` or `classification/multi_stage_train.py`. By default, they will use the conventional softmax.
1. Extract the base model weights using `classification/save_base_model.py`.
1. Train a model by specifying --use_bags and also passsing the pretrained weights using --base_model_weights option. See `classification/configs/efficientnet_b2_multicrop_bags_final_submission_training.config`.

### Prediction

#### Counting heuristic

For our final solution, we followed the competition benchmark counting heuristic: the maximum number of bounding boxes across any image in the sequence. We only count bounding boxes with confidence > 0.8.

This counting strategy limits us to predict only one species per sequence. We see our solution as a strong baseline for counting animals on camera trap sequences, but we do believe that the best solution should be tracking animals across images (Multi-object tracking), classify each track, and count them. We tried DeepSORT to track animals, but it wasn't as good as this heuristics for counting (see the next section).

#### Classification

The prediction for each image is a weighted average of the predictions for the bounding boxes with the highest score and the full image (0.15 full image + 0.15 full image (flip) + 0.35 bbox + 0.35 bbox (flip)) as used by the winning solution of iWildCam 2020. To classify a sequence, we average the predictions of all nonempty images for each sequence.

#### Final submission

Both bbox and full image models of our final submission are based on EfficientNet-B2 using Balanced Group Softmax to handle the class imbalance in the dataset (see `classification/configs` folder). The bbox model was trained using all bounding boxes with confidence > 0.6. We run MegaDetector V4 to generate bounding boxes. The trained models used to generate predictions can be found [here](https://drive.google.com/drive/folders/1jM_U0tyvYr0aRsZI_g3aUwYNAKPqqWl1?usp=sharing) and the MegaDetector V4 bboxes for iWildCam 2021 are available [here](https://drive.google.com/drive/folders/19LNFfvEVmOf1NDsT0jn4v47socpFbXek?usp=sharing).

Use the script `classification/predict_bbox_n_full.py` to generate a submission:
```bash
python predict_bbox_n_full.py --annotations_json=PATH_TO_BE_CONFIGURED/iwildcam2021_train_annotations.json \
    --dataset_dir=PATH_TO_BE_CONFIGURED/ \
    --megadetector_results_json=PATH_TO_BE_CONFIGURED/iwildcam2021test_originalimage_megadetector_v4.1_results_parsed.json \
    --test_info_json=PATH_TO_BE_CONFIGURED/iwildcam2021_test_information.json \
    --submission_file_path=PATH_TO_BE_CONFIGURED/final_submission.csv \
    --model_name=efficientnet-b2 \
    --use_bags \
    --batch_size=16 \
    --input_size=380 \
    --input_scale_mode=uint8 \
    --ckpt_dir_full=PATH_TO_BE_CONFIGURED/fixefficientnet_b2_380x380_iwildcam_fulltrain_mdv4_fullimage_16mai_bags_mltstg/ \
    --ckpt_dir_bbox=PATH_TO_BE_CONFIGURED/fixefficientnet_b2_380x380_iwildcam_fulltrain_mdv4_multicrop_26mai_bags_mltstg/ \
    --use_flip_image \
    --ensemble_method=averaging \
    --megadetector_threshold=0.8
```

### Other things that we tried

#### GPS coordinates

We tried to use the [Geo Prior model](https://arxiv.org/abs/1906.05272) to improve prediction using GPS coordinates and time of year, but it did not work with the model overfitting the training set. We believe GPS coordinates can be useful for the problem (it worked very well for our [iNat 2021 solution](https://github.com/alcunha/inat2021ufam)), but it's necessary to develop a model to deal with camera trap specificities such as the fixed position.

This repository includes code to predict using a trained geo prior model that can be trained using our [Tensorflow implementation](https://github.com/alcunha/geo_prior_tf/). See the script `classification/eval_main.py` for predictions combined with geo priors. The Geo Prior model doesn't learn using the original loss, so we had to replace it with the focal loss. But using only the classifier predictions was better than using them combined with geo priors.

#### DeepSORT to track animals

We tried to use [DeepSORT](https://arxiv.org/abs/1703.07402) to generate tracks over sequences. Then we classify each track by averaging the predictions of its bounding boxes list. We use the feature embedding from EfficientNet-B2 trained on bounding boxes. We also tried to use features from ReID models trained by the [winner of ECCV TAO 2020 challenge](https://github.com/feiaxyt/Winner_ECCV20_TAO), but they performed slightly worst than using EfficientNet-B2 features.

To extract features using EfficientNet-B2, use the script `mot/generate_features.py`:
```bash
python generate_features.py --model_name=efficientnet-b2 \
    --input_size=380 \
    --input_scale_mode=uint8 \
    --base_model_weights=PATH_TO_BE_CONFIGURED/efficientnet_b2_crop_25mai.h5 \
    --test_info_json=PATH_TO_BE_CONFIGURED/iwildcam2021_test_information.json \
    --dataset_dir=PATH_TO_BE_CONFIGURED \
    --megadetector_results_json=PATH_TO_BE_CONFIGURED/iwildcam2021test_originalimage_megadetector_v4.1_results_parsed.json \
    --min_confidence=0.9 \
    --random_seed=42 \
    --features_file=PATH_TO_BE_CONFIGURED/efficientnet_b2_crop_25mai_features.json
```

To track animals with DeepSORT use the script `track_iwildcam.py` from our [DeepSORT repository](https://github.com/alcunha/deep_sort_iwildcam2021ufam). We kept DeepSORT code on a separate repository to avoid GPLv3 licensing conflicts.

Finally, to classify tracks and generate a submission, use the script `classification/predict_track.py`:
```bash
python predict_track.py --annotations_json=PATH_TO_BE_CONFIGURED/iwildcam2021_train_annotations.json \
    --dataset_dir=PATH_TO_BE_CONFIGURED \
    --megadetector_results_json=PATH_TO_BE_CONFIGURED/iwildcam2021test_originalimage_megadetector_v4.1_results_parsed.json \
    --test_info_json=PATH_TO_BE_CONFIGURED/iwildcam2021_test_information.json \
    --submission_file_path=PATH_TO_BE_CONFIGURED/deepsorttracks.csv \
    --model_name=efficientnet-b2 \
    --use_bags \
    --batch_size=32 \
    --input_size=380 \
    --input_scale_mode=uint8 \
    --ckpt_dir=PATH_TO_BE_CONFIGURED/fixefficientnet_b2_380x380_iwildcam_fulltrain_mdv4_multicrop_26mai_bags_mltstg/ \
    --tracks_file=PATH_TO_BE_CONFIGURED/efficientnet_b2_crop_25mai_tracks.json \
    --num_images_by_track=8
```

### Contact

If you have any questions, feel free to contact Fagner Cunha (e-mail: fagner.cunha@icomp.ufam.edu.br) or Github issues. 

### License

[Apache License 2.0](LICENSE)