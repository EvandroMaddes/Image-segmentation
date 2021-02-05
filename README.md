# Image-segmentation
https://competitions.codalab.org/competitions/27176
REPORT SECOND CHALLENGE: SEMANTIC SEGMENTATION 

We started from the notebook for multi-class segmentation seen at the exercise session with prof. Lattari. We changed the part of code in order to deal with a three classes segmentation problem, in particular we modified the generation of the custom datasets for both the training and the validation set. We added code in order to generate two text files where to write the splitting between training and validation. We also defined a new dataset for the test set. 
We focused on the Bipbip dataset using both images from Mais and haricot. We tried also to use both Bipbip and Weedelec dataset together, but the performances did not increase as expected.  
We then applied some basic data augmentation: horizontal flip, weight and height shift, zoom, rotation. 
We spilt the data in training 80% and validation set 20%. 
After that, we plotted the images to see them with their corresponding masks and we defined the generators for each set, not augmenting the validation set. 
Starting from the model of the prof. Lattari, we tuned parameters. We worked mainly on the learning rate and on the image size. The main problem of this network is the overfitting, so we reduced the learning rate (1e^-5) and we increased the input size up to 512x512. Changing also values of the parameters of image generator allowed us to overcome overfitting. 
However, this model did not reach very high score (no more than 0.3 on the global IoU). 
Then we moved to a transfer learning approach in which we adopted different models for encoding and decoding.  
Model available as encoder:  

Models available as decoder: Unet, FPN, Linknet, PSPNet 
We worked in parallel on four models: one for each dataset. 
Since we noticed no big differences for images of Bipbip and Weedelec datasets, we adopted similar models for these two datasets. 
For both models the decoder is the “efficientnetb4” (in the notebook we link to the specific documentation). This backbone is chosen due to the low number of parameters. Since we have few data for training, we decided to choose a simple model wrt parameters.  
The decoder is a feature pyramid network pre-trained on ImageNet. 
The preprocess function is the one of the efficient net, in particular in the code when we set the encoder model, we also set the relates preprocess function. 
For Bipbip model, the image size is 768x1024 in order to keep the same proportion as in the original dataset. 
For Weedelec model, the image size is 512x512. 
The batch size is set to one because higher value leads to an error in training phase (the colab notebook run out of memory). 
After the generation of the three sets (training, validation and test) we plotted the images from validation and training sets in order to see the effect of the preprocess function. 
We used different learning rate: 1e^-5 for the Bipbip model while 3e^-5 for the Weedelec model. 
Each model trains on one specific dataset using as metrics Intersection of Union. 
For what concerns Pead and Roseau, we used “Vgg16” combined with FPN, with a learning rate set to 1e-5, the sparse categorical cross-entropy and Adam as optimizer. We only used horizontal flip and shifts for augmentation, with image size 512x512 and batch size 1. 
During this challenge we also try to approach the problem in different way. We did our best to follow the advice of professor Matteucci, implementing the tiling of the images. We were able to implement the function to split an image in tiles and the respective mask. For what concerns the reconstruction of the image we encounter some trouble and we decided to give up. 
Another approach was to split the data based on the seed (mais, haricot) but the results obtained did not improve the previous one.	 
For all the models, during the training, we used early stopping and reduce on plateau functions to get the best results. We also set a scheduler for the learning rate based on the number of epochs but at the end we did not use it. The value of parameters of these functions are mainly based on the current learning rate and on the number of parameters of the entire model. 
After training we plotted the images from the validation set, the correspond mask and the predicted by our model in order to evaluate the weakness points of the model. 
Finally, we achieved good results in terms of generalization; monitoring the mean IoU on the validation set, we did not encounter overfitting until 0,5. From 0,6 on we started to overfit even if the training ended just after. 

 

