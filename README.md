# Goals
1. Write a function that extracts the frames referenced by the xml files. We recommend extracting the frames at some lower resolution that's more manageable for ML purposes.
2. Extract a sample of frames that are "far enough" away from frames that have positive labels. These frames can be your negative samples.
3. Do data augmentation to get 10x the data with the goal of increasing data diversity for ML purposes.
4. Create a classifier to detect whether there's a polyp in the frame

## Training Data
* Videos 6.mp4 through 11.mp4 contain training examples
* Files 6.xml through 11.xml contain bounding box annotations
* clean.mp4 does not have annotations

# Final solution
## Extract Frames
The data curation includes extracting the frames from the videos and drawing bounding boxes.
* to run: python curate_data.py

The images will save positive frames to /data/positive, negative frames to /data/negative, and the positive samples with bounding boxes drawn to /boxes/. It also saves frames from the clean video file to a /test/ folder. Training images are saved at a lower resolution, 1/4 of the original image size. You have the ability to adjust the threshold for distance between positive and negative samples (in frames) and the frequency at which the frames save.

See the code for more comments and implementation details.

## Bounding boxes
Here are some examples of bounding boxes:
![bounding box example 1](/images/pos_10_frame_450.jpg)
![bounding box example 2](/images/pos_10_frame_502.jpg)
Although these are not exactly aligned, it appears that the script is correctly matching the box coordinates to each frame. This may be an issue with the frame rate and using OpenCV to iterate through each frame. Currently the bounding boxes are only saved as one per frame, even if there are multiple polyps in a frame (multiple images will save for that frame with one box each). The next iteration of this work would add the ability to save with multiple bounding boxes.

## Data augmentation
Data augmentation is done using transforms in PyTorch. Each image is scaled and cropped in the center to 224x224 pixels. There is a random horizontal flip 50% of the time, and the image is normalized and converted to a tensor to be used in PyTorch. More complex data augmentation to inflate the size of the training set could be done with the imgaug package or basic numpy if given more time.

## Neural network training
A binary classifier was trained to identity whether a frame contains a polyp. We run it for 40 epochs and then fine tune to training for another 40 epochs.

* to run: python train.py --train_dir ~/train--val_dir ~/vali --test_dir ~/test --use_gpu

Here are the results:\
Starting epoch 29 / 40\
Train accuracy:  1.0\
Val accuracy:  0.9692307692307692

Starting epoch 30 / 40\
Train accuracy:  1.0\
Val accuracy:  0.9807692307692307

Starting epoch 31 / 40\
Train accuracy:  1.0\
Val accuracy:  0.9846153846153847

Starting epoch 32 / 40\
Train accuracy:  1.0\
Val accuracy:  0.9923076923076923

As you can see, training and validation accuracy are quite high. This was with a 80/20 split of training to validation data with about 1000 total images from the training videos. Here are some test results:

Test 1: using frames from clean.mp4 as test\
Test accuracy : 0.7015113350125944\
I suspect the results are poor since I manually decided whether frames contained polyps or not, and I probably did a poor job of it. Some of them were ambiguous to me even as a human verifier.

Test 2: using random frames from videos 6-11 but making sure these are unique from the training data\
Test accuracy : 0.9751243781094527\
This accuracy is suspiciously high, and I realized it is because similar frames could be spread between training and test data (given the sequence of related frames in a video).

Test 3: using frames from video 11, which were removed from the train and validation sets\
Test accuracy : 0.7844311377245509\
Saving video 11 for test data ensure that the data was labeled and I didn't need to annotate them myself. This accuracy is still not great, and given a train accuracy of 100% and validation accuracy of 98.75% for this round, it appears that the model is overfitting. Some ways to combat this are using regularization or dropout layers.

For retesting with the same previously trained model, use test.py
* to run: python test.py --test_dir ~/data/test

This takes the saved model (classifier.pth) and runs it on the new test data. The previously trained models are saved in the /models/ folder.
