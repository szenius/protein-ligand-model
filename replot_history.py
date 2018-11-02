from utils import load_pickle, plot_single_data

model_name = 'Dual-stream 3D Convolution Neural Network'
# model_name = 'Baseline 5x256 MLP'

history = load_pickle('history.pkl')

plot_single_data(
    history['loss'],
    'b',
    'epoch',
    'loss',
    '{} Train Loss'.format(model_name),
    '{} Train Loss.png'.format(model_name)
)

plot_single_data(
    history['acc'],
    'r',
    'epoch',
    'acc',
    '{} Train Accuracy'.format(model_name),
    '{} Train Accuracy.png'.format(model_name)
)