# Protein-Ligand Binding Affinity Prediction
<INTRO FROM REPORT HERE>

# Dataset preparation

## Training Data
Training data should be placed in the directory `training_data/`, and this directory should reside in the project root directory. 

The format of the training data files, be it naming or content, are as per what was given to us.

## Testing Data
Similarly, testing data should be placed in the directory `testing_data/`, and this directory should reside in the project root directory.

The format is also as per what was given to us.

# Training

The following assumes that you are using Python3 and have libraries like `numpy` and `keras` installed. You may also need to install [GraphViz](https://anaconda.org/conda-forge/python-graphviz) as the training step also tries to print out a visualization of the model. 

For both models, you should expect the following output:
- A graph with loss and accuracy plots saved as a `.png` file
- Model weights saved as a `.h5` file
- Model visualization saved as a `.png` file

## Dual-stream 3D Convolution Neural Network
To start training the Dual-stream 3DCNN, simply run the following command.

```
python train.py
```

## Baseline 5x25 MLP
To start training the MLp model, simply run the following command.

```
python dist_train.py
```

# Prediction

First make sure you change the `WEIGHTS_FILENAME` variable in `predict_utils.py`. Then run the following command.

```
python predict.py
```

The predictions of the top ten ligand candidates for binding to any protein will be printed in the file `test_predictions.txt`.