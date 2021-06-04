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

### Other things that we tried

#### GPS coordinates

We tried to use the [Geo Prior model](https://arxiv.org/abs/1906.05272) to improve prediction using GPS coordinates and time of year, but it did not work with the model overfitting the training set. We believe GPS coordinates can be useful for the problem (it worked very well for our [iNat 2021 solution](https://github.com/alcunha/inat2021ufam)), but it's necessary to develop a model to deal with camera trap specificities such as the fixed position.

This repository includes code to predict using a trained geo prior model that can be trained using our [Tensorflow implementation](https://github.com/alcunha/geo_prior_tf/). See the script `classification/eval_main.py` for predictions combined with geo priors. The Geo Prior model doesn't learn using the original loss, so we had to replace it with the focal loss. But using only the classifier predictions was better than using them combined with geo priors.

#### DeepSORT to track animals

### Contact

If you have any questions, feel free to contact Fagner Cunha (e-mail: fagner.cunha@icomp.ufam.edu.br) or Github issues. 

### License

[Apache License 2.0](LICENSE)