# 3D-pose-estimation

This project aims to train a 3D pose estimation model based on a given `.obj` file. It is devided into two steps:

- Generation of a synthetic dataset from the `.obj` file 

- Training of a ResNet-50 to predict the pose based on an image showing a specfic perspective

## Synthetic Dataset Generation

The dataset generator will rotate the object randomly and create a directory `images` containing the different views and a file `labels.txt` with the three Euler angles used for the rotation. The following prompt will create 5000 datapoints for the given object:

`python create_dataset.py --n_samples 5000 --file "WOLF.OBJ"`

## Training

To train a ResNet-50 pretrained on ImageNet, use the following promt:

`python main.py`

By default, the model is trained with a batch norm of 64, learning rate of 5e-5, 100 epochs and saved as `saved_model.pt`.

You can specify the following parameters:
- `--gpu`: GPU ID (int)
- `--bn`: batch size (int)
- `--lr`: learning rate (float)
- `--epochs`: number of training epochs (int)
- `--euler`: whether euler angles should be used as predictions instead of 6D representations (store_true)
- `--rgb`: whether rgb images should be used for training instead of binary (store_true)
- `--save_name`: name of the saved model (str)

## Inference

You can test your model using the file `inference.py`. Use the following prompt to predict the rotation of a specific image:

`python inference.py --input {your_image} --save_img --save_matrix`

This will save the an image of the rotated object as `prediction.png` and the corresponding rotation matrix as `prediction.txt`. You can specify the following parameters:

- `--input`: name of the input image (str)
- `--load_name`: name of the saved model file (str)
- `--save_img`: whether to save the rotated object as image (store_true)
- `--save_matrix`: whether to save the rotation matrix as text file (store_true)
- `--euler`: whether to use Euler angles for prediction (store_true)
- `--rgb`: whether to use the rgb image instead of binary for the prediction (store_true)
- `--example`: whether to use the example image and pre-trained model provided in this repo (store_true)
