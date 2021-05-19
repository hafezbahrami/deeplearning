FCN design
Design your FCN by writing the model in models.py. Make sure to use only convolutional operators, pad all of them correctly and match strided operators with up-convolutions. Use skip and residual connections.

Make sure your FCN handles an arbitrary input resolution and produces an output of the same shape as the input. Use output_padding=1 if needed. Crop the output if it is too large.

Test your model with

python3 -m grader homework
FCN Training
To train your FCN you'll need to modify your CNN training code a bit. First, you need to use the DenseSuperTuxDataset. This dataset accepts a data augmentation parameters transform. Most standard data augmentation in torchvision do not directly apply to dense labeling tasks. We thus provide you with a smaller subset of useful augmentations that properly work with a pair of image and label in dense_transforms.py.

You will need to use the same bag of tricks as for classification to make the FCN train well.

Since the training set has a large class imbalance, it is easy to cheat in a pixel-wise accuracy metric. Predicting only track and background gives a 96% accuracy. We additionally measure the Intersection-over-Union evaluation metric. This is a standard semantic segmentation metric that penalizes largely imbalanced predictions. This metric is harder to cheat, as it computes  \frac{\text{true positives}}{\text{true positives} + \text{false positives} + \text{false negatives}}.  You might need to change the class weights of your torch.nn.CrossEntropyLoss, although our master solution did not require this. You can compute the IoU and accuracy using the ConfusionMatrix class.

Test your model with

python3 -m grader homework
Relevant Operations
torch.optim.Adam might train faster
torch.nn.ConvTranspose2d
torch.cat for skip connections
and all previous
Grading
The test grader we provide

python3 -m grader homework -v
will run a subset of test cases we use during the actual testing. The point distributions will be the same, but we will use additional test cases. More importantly, we evaluate your model on the test set. The performance on the test grader may vary. Try not to overfit to the validation set too much.

Submission
Once you finished the assignment, create a submission bundle using

python3 bundle.py homework [YOUR UT ID]
and submit the zip file on canvas. Please note that the maximum file size our grader accepts is 20MB. Please keep your model compact. Please double-check that your zip file was properly created, by grading it again

python3 -m grader [YOUR UT ID].zip
Online grader
We will use an automated grader through canvas to grade all your submissions. There is a soft limit of 5 submisisons per assignment. Please contact the course staff before going over this limit, otherwise your submission might be counted as invalid.

The online grading system will use a slightly modified version of python and the grader:

Please do not use the exit or sys.exit command, it will likely lead to a crash in the grader
Please do not try to access, read, or write files outside the ones specified in the assignment. This again will lead to a crash. File writing is disabled.
Network access is disabled. Please do not try to communicate with the outside world.
Forking is not allowed!
print or sys.stdout.write statements from your code are ignored and not returned.
Please do not try to break or hack the grader. Doing so will have negative consequences for your standing in this class and the program.

Running your assignment on google colab
You might need a GPU to train your models. You can get a free one on google colab. We provide you with a ipython notebook that can get you started on colab for each homework.

If you've never used colab before, go through colab notebook (tutorial)
When you're comfortable with the workflow, feel free to use colab notebook (shortened)

Follow the instructions below to use it.

Go to http://colab.research.google.com/.
Sign in to your Google account.
Select the upload tab then select the .ipynb file.
Follow the instructions on the homework notebook to upload code and data.

Honor code
This assignment should be solved individually.

What interaction with classmates is allowed?

Talking about high-level concepts and class material
Talking about the general structure of the solution (e.g. You should use convolutions and ReLU layers)
Looking at online solutions, and pytorch samples without directly copying or transcribing those solutions (rule of thumb, do not have your coding window and the other solution open at the same time). Always cite your sources in the code (put the full URL)!
Using any of your submissions to prior homework
Using the master solution to prior homework
Using ipython notebooks from class
What interaction is not allowed?

Exchange of code
Exchange of architecture details
Exchange of hyperparameters
Directly (or slightly) modified code from online sources
Any collaboration
Putting your solution on a public repo (e.g. github). You will fail the assignment if someone copies your code.
Ways students failed in past years (do not do this):

Student A has a GPU, student B does not. Student B sends his solution to Student A to train 3 days before the assignment is due. Student A promises not to copy it but fails to complete the homework in time. In a last-minute attempt, Student A submits a slightly modified version of Student B's solution. Result: Both students fail the assignment.
Student A struggles in class. Student B helps Student A and shows him/her his/her solution. Student A promises to not copy the solution but does it anyway. Result: Both students fail the assignment.

Student A sits behind Student B in class. Student B works on his homework, instead of paying attention. Student A sees Student B's solution and copies it. Result: Both students fail the assignment.

Student A and B do not read the honor code and submit identical solutions for all homework. Result: Both students fail the class.

Installation and setup
Installing python 3
Go to https://www.python.org/downloads/ to download python 3. Alternatively, you can install a python distribution such as Anaconda. Please select python 3 (not python 2).

Installing the dependencies
Install all dependencies using

python3 -m pip install -r requirements.txt
Note: On some systems, you might be required to use pip3 instead of pip for python 3.

If you're using conda use

conda env create environment.yml
The test grader will not have any dependencies installed, other than native python3 libraries and libraries mentioned in requirements.txt. This includes packages like pandas. If you use additional dependencies ask on piazza first, or risk the test grader failing.

Manual installation of pytorch
Go to https://pytorch.org/get-started/locally/ then select the stable Pytorch build, your OS, package (pip if you installed python 3 directly, conda if you installed Anaconda), python version, cuda version. Run the provided command. Note that cuda is not required, you can select cuda = None if you don't have a GPU or don't want to do GPU training locally. We will provide instruction for doing remote GPU training on Google Colab for free.

Manual installation of the Python Imaging Library (PIL)
The easiest way to install the PIL is through pip or conda.

python3 -m pip install -U Pillow
There are a few important considerations when using PIL. First, make sure that your OS uses libjpeg-turbo and not the slower libjpeg (all modern Ubuntu versions do by default). Second, if you're frustrated with slow image transformations in PIL use Pillow-SIMD instead:

CC="cc -mavx2" python3 -m pip install -U --force-reinstall Pillow-SIMD
The CC="cc -mavx2" is only needed if your CPU supports AVX2 instructions. pip will most likely complain a bit about missing dependencies. Install them, either through conda, or your favorite package manager (apt, brew, ...).

Grading
100.0 points possible
Please upload this assignment through canvas.

