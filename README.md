# Stammer-Detection-model
A speech disorder known as stuttering is characterised by the repetition of sounds, syllables, or words; prolongation of sounds; and blocks—interruptions in speech. A person who stutters is fully aware of what they want to say but struggles to deliver it in a natural manner. These pauses in speech may be accompanied by signs of struggle, like rapid blinking of the eyes or trembling of the lips. Communication difficulties brought on by stuttering frequently affect a person's quality of life and interpersonal relationships. Additionally, stuttering can have a negative impact on opportunities and job performance, and its treatment can be very expensive. That is why we are making this project where a person stutters the machine corrects the stammer and then it speaks fluently without any pause or stutter. So basically we will be using apart of artificial intelligence that is machine learning . So what we will do in this project is we will first convert the speech to text now we will tale this text and where ever we see some pauses or if for example a letter is repeated many times then it will be recognized as stutter then the corrected text will then be converted to speech. After this we will try to integrate entire project to website and then as end result we will have a website that helps people who stammer. Most of the project will be done using python language. But HTML(hyper text markup language), CSS(cascading style sheets), Javascript, Nodejs, expressjs etc will also be required.


## How it works

### Preprocessing

Firstly we have loaded all the audio files using pydub library using AudioSegment.from_file
then we converted that audio files into array using np getArrayOfSamples()
Then applied noise reduce function to reduce the noise
and then we use os to save file into another directory where all the wav file have noise removed

Next we need to convert them into spectographs
now change the extionssion from .wav to .png using split path
After that we check if the wav file has spectogram generate or not is not then we use matplot lib to generate spectogram using plot.spectogram()


### Building Neural Network
We start by cropping unwanted part of data done using layers of keras library
image is resized into 256 x 256 pixels
Images are normalized into [0,1] to ensure pixel values are within specific range

Convolutional Layers:
The model has three convolutional layers (Conv2D), each followed by batch normalization (BatchNormalization) to accelerate training and improve convergence.
The first convolutional layer has 32 filters with a kernel size of 3x3, a stride of 2, and 'swish' activation function. Swish is a type of activation function that has been found to perform well in deep neural networks.
The second convolutional layer also has 32 filters with a kernel size of 3x3, a stride of 2, and 'swish' activation function.
The third convolutional layer has 64 filters with a kernel size of 3x3 and 'swish' activation function.

Max Pooling Layers: Two max-pooling layers (MaxPooling2D) are used to downsample the feature maps obtained from convolutional layers. Max pooling reduces the spatial dimensions of the feature maps, helping in capturing the most important features while reducing computational complexity.
The first max-pooling layer uses a pool size of 2x2.
The second max-pooling layer also uses a pool size of 2x2

Flattening Layer: The output of the last convolutional layer is flattened using the Flatten layer, converting the 2D feature maps into a 1D vector. This prepares the data for input to the fully connected layers.

Fully Connected Layers:
The flattened output is passed through two fully connected (Dense) layers with 512 and 128 units, respectively. These layers introduce non-linearity to the model and learn higher-level features from the extracted image features.
Each fully connected layer is followed by batch normalization (BatchNormalization) to improve training stability and convergence.
Dropout Layer: A dropout layer (Dropout) is added with a dropout rate of 0.5. Dropout randomly sets a fraction of input units to zero during training, which helps prevent overfitting by forcing the model to learn more robust features.

Output Layer: The output layer consists of two units with softmax activation function, which outputs the probability distribution over the two classes (stuttering and non-stuttering). Softmax ensures that the output probabilities sum up to one, making it suitable for multi-class classification.



### Compilation and Training
compiles the model using the compile() method, specifying the loss function (categorical_crossentropy), optimizer (RMSprop), and metrics to monitor during training.
trains the model using the fit() method, passing the training dataset and specifying the number of epochs
ves the trained model to a file named "CNN_model.h5" using the save() method.


### Result
The score was evaluated which gave accuracy of about 80%
