from utils import load_pickle, plot_single_data, plot

model_name = 'Dual-stream 3D Convolution Neural Network'
# model_name = 'Baseline 5x256 MLP'

'''
Single model
'''
history = load_pickle('conv/history.pkl')

print(history['loss'][-1], history['acc'][-1])

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


'''
Batch size comparison
'''

history1 = load_pickle('conv batch 32 epoch 10 dim 10/history.pkl')
history2 = load_pickle('conv batch 64 epoch 10 dim 10 ratio 1/history.pkl')
history3 = load_pickle('conv batch 128 epoch 10 dim 10/history.pkl')

plot(
    [history1['acc'], history2['acc'], history3['acc']],
    ["batch_size 32", "batch_size 64", "batch_size 128"],
    ['c', 'b', 'k'],
    "epoch",
    "accuracy",
    model_name + " Train Accuracy (10x10x10, 1:1)",
    "batch_size_acc.png"
)


'''
Dimensions
'''

history1 = load_pickle('conv batch 128 epoch 10 dim 10/history.pkl')
history2 = load_pickle('conv batch 128 epoch 10 dim 15/history.pkl')
history3 = load_pickle('conv batch 128 epoch 10 dim 25/history.pkl')

plot(
    [history1['acc'], history2['acc'], history3['acc']],
    ["10x10x10", "15x15x15", "25x25x25"],
    ['y', 'm', 'r'],
    "epoch",
    "accuracy",
    model_name + " Train Accuracy (128, 1:1)",
    "dim_acc.png"
)



'''
Ratio
'''
history1 = load_pickle('conv batch 128 epoch 10 dim 10/history.pkl')
history2 = load_pickle('conv batch 128 epoch 10 dim 10 ratio 3/history.pkl')
history3 = load_pickle('conv batch 128 epoch 10 dim 10 ratio 5/history.pkl')

plot(
    [history1['loss'], history2['loss'], history3['loss']],
    ["1:1", "1:3", "1:5"],
    ['y', 'b', 'g'],
    "epoch",
    "loss",
    model_name + " Train Loss (128, 10x10x10)",
    "ratio_loss.png"
)