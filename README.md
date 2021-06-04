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

### Prediction

### Other things that we tried

#### GPS coordinates

We tried to use the [Geo Prior model](https://arxiv.org/abs/1906.05272) to improve prediction using GPS coordinates and time of year, but it did not work with the model overfitting the training set. We believe GPS coordinates can be useful for the problem (it worked very well for our [iNat 2021 solution](https://github.com/alcunha/inat2021ufam)), but it's necessary to develop a model to deal with camera trap specificities such as the fixed position.

This repository includes code to predict using a trained geo prior model that can be trained using our [Tensorflow implementation](https://github.com/alcunha/geo_prior_tf/). The model doesn't learn using the original loss, so we had to replace it with the focal loss. But using only the classifier predictions was better than using them combined with geo priors.

#### DeepSORT to track animals

### Contact

If you have any questions, feel free to contact Fagner Cunha (e-mail: fagner.cunha@icomp.ufam.edu.br) or Github issues. 

### License

[Apache License 2.0](LICENSE)