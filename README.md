# A-NET
It is a Siamese Network.

## Siamese Network
A Siamese neural network is an artificial neural network that uses the same weights while working in tandem on two different input vectors to compute comparable output vectors. Often one of the output vectors is precomputed, thus forming a baseline against which the other output vector is compared.

![image](https://user-images.githubusercontent.com/44013285/156606423-44b16f42-c42b-4a54-9508-3bc368f96ed5.png)

## Siamese Function(L1-Distance)
```python
class L1Distance(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
```
As the L1Distance decrease our model gaining accuracy.

Total Model Parameters: ```27,430,209```

## Optimizer and Loss Function
```python
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)
```
```
I have gained accuracy of 100% on Face Recognition.
```
