# Coursework PART I - Shape Detection

> This part of the coursework requires [Python 3.6](https://www.python.org/downloads/).
> For Windows, you might want to use [Conda](https://www.anaconda.com/products/distribution). 

## Marking Scheme 
This part is worth 50% of the CW mark for the unit. Marks will be allocated as follows. For subtasks 1-3, marks will be based on the content of the report w.r.t description, analysis and evaluation and on the code.
1. Subtask 1: The Dartboard Detector - 15%
2. Subtask 2: Integration with Shape Detectors - 15%
3. Subtask 3: Improving your Detector - 10%
4. Overall standard of presentation, analysis and evaluation - 10%

 
## Introduction
Detecting (locating & classifying) instances of an object class in images is an important application in computer vision as well as an ongoing area of research. This assignment requires you 1) to experiment with the classical Viola-Jones object detection framework as discussed in the lectures and provided by OpenCV, and 2) to combine it with other detection approaches to improve its efficacy. Your approach is to be tested and evaluated on a small image set that depicts aspects of an important traffic sign – Dartboard.

## :red_circle: Subtask 1: The Dartboard Detector
_(15 marks)_

This subtask requires you to build an object detector that recognises dartboards. The initial steps of this subtask introduce you to OpenCV’s boosting tool, which you can use to construct an object detector that utilises Haar-like features. Training the boosted cascade of weak classifiers works with OpenCV 3.4 which requires Python 3.6. If you use Mac M1/M2, we strongly recommend you train your detector on the lab machine, and then you may transfer your model (cascade.xml) to work on your own machine.

1. For the machine in Lab2.11, load conda with `module load anaconda/3-2023`. Create virtual environment with Python 3.6 `conda create -n ipcv36 python=3.6`, activate your environment `conda activate ipcv36`, and install OpenCV packages `conda install -c menpo opencv`. Check OpenCV verion with `python -c 'import cv2; print(cv2.__version__)'`. It should be 3.4.x. 
   
   <details>
    <summary> Python 3.6 Installation Troubleshooting </summary>
    
     > 1. Download installer from https://www.python.org/downloads/release/python-368/
     > 2. Create virtual environment with `python3.6 -m venv ipcv36` (if doesn't work, try `python3 -m venv ipcv36`), then activate your environment `source ipcv36/bin/activate`
   </details>

   <details>
   <summary> OpenCV 3.4 for Windows Troubleshooting </summary>
    
     > 1. Download OpenCV3.4.3 from [HERE](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.3/). 
     > 2. Extract it to `YOURPATH`
   </details>

   <details>
    <summary> OpenCV 3.4 for macOS Troubleshooting </summary>
    
     > If you use an Intel-based Mac, you should not have a problem of installing Python 3.6 and OpenCV 3.4. 
     >
     > If you use an ARM-based system, please try with Conda version before 23.0 (e.g., 22.9.0). If it still does not work, please use the Lab machine in 2.11.
     > 
  </details>

2. You are given `dart.bmp` containing a dartboard that can serve as a prototype for generating a whole set of positive training images. 
3. Unzip `negatives.zip` and keep all negative images in a directory called `negatives`. A text file `negatives.dat` lists all filenames in the directory.
4. Create your positive training data set of 500 samples of dartboards from the single prototype image provided. To do this, you can run the tool `opencv_createsamples` via the following single command and execute it in a folder that contains the negatives.dat file, the dart.bmp image and the negatives folder: 

```
opencv_createsamples -img dart.bmp -vec dart.vec  -w 20 -h 20 -num 500 -maxidev 80 -maxxangle 0.8 -maxyangle 0.8 -maxzangle 0.2
```
For Windows, you might use `YOURPATH\opencv\build\x64\vc15\bin\opencv_createsamples.exe`.

5. This will create 500 tiny 20×20 images of dartboards (later used as positive training samples) and store in the file `dart.vec`, which contains all these 500 small sample images. Each of the sample images is created by randomly changing viewing angle and contrast (up to the maximum values specified) to reflect the possible variability of viewing parameters in real images better.
6. Now use the created positive image set to train a dartboard detector via AdaBoost. To do this, create a directory called `Dartboardcascade` in your working directory. Then run the `opencv_traincascade` tool with the following parameters as a single command in your working directory:
```
opencv_traincascade -data Dartboardcascade -vec dart.vec -bg negatives.dat -numPos 500 -numNeg 500 -numStages 3 -maxDepth 1 -w 20 -h 20 -minHitRate 0.999  -maxFalseAlarmRate 0.05 -mode ALL
```
For Windows, you might use `YOURPATH\opencv\build\x64\vc15\bin\opencv_traincascade.exe`

7. This will start the boosting procedure and construct a strong classifier stored in the file `cascade.xml`, which you can load in an OpenCV program for later detection as done in Lab4: Face Detection (`face.py`). You might need the change `model = cv2.CascadeClassifier()` to `model = cv2.CascadeClassifier(cascade_name)` or remove `cv2.samples.findFile`.
8. During boosting the tool will provide updates about the machine learning in progress. Here is an example output when using 1000 instead of 500 samples…
<img src="https://github.com/UoB-CS-IPCV/CW-I-Shape-Detection/blob/main/trainresult.png" height=200> 
The boosting procedure considers all the positive images and employs sampled patches from the negative images to learn. The detector window will be 20×20. To speed up the detection process, the strong classifier is built in 3 parts (numStages) to form an attentional cascade as discussed in the Viola-Jones paper. The training procedure may take up to 5min for reasons discussed in the lectures – stop the training and restart if it exceeds this time. If the training procedure exits with "Required leaf false alarm rate achieved. Branch training terminated.", please use the lab machine in Lab2.11.

### What to say in your report (roughly 1 page): 

a)	TRAINING PERFORMANCE: The training tool produces a strong classifier in stages. Per stage the tool adds further features to the classifier and prints the achieved TPR and FPR (false positive rate) for that point on the training data (see Figure above). 
1. Collate this information into a graph that plots TPR and FPR on the training data for the three different stages. 
2. Produce this graph in your report and briefly interpret what it shows.

b)	TESTING PERFORMANCE: In face.py from Lab4, change "frontalface.xml" to "Dartboardcascade/cascade.xml". Now, you may go back to use Python 3.8.
1. Test the dartboard detector’s performance on all given example images. 
2. Produce the result images with bounding boxes drawn around detected dartboard candidates (in green) and ground truth (in red) and include 3 of them in your report.  
3. In tabular form, calculate the overall TPR and F1 score per image and the average of these scores across the 16 images. 
4. Briefly discuss the performance achieved.
5. Give reasons for the different TPR values compared to the performance achieved in a).

## :red_circle: Subtask 2: Integration with Shape Detectors
_(15 marks)_

For the second subtask, use at least one Hough Transform on the query images in order to detect important shape configurations of dartboards. Feel free to use and/or adapt your implementation of the Hough Transform from former formative tasks. You must implement your own Hough transform(s). You may reuse your code from Lab 3: Coin Counter Challenge. Utilize the information (e.g. on lines, circles, ellipses) to aid dartboard detection.  Think carefully how to combine this new evidence with the output of your Viola-Jones detector in order to improve on results. Implement and evaluate this improved detector.

### What to say in your report (roughly 1 page): 

a)	HOUGH DETAILS: Show in your report for two of the given example dartboard images which best exhibit the merit and limitations of your implementation: 
1. The thresholded gradient magnitude image used as input to the Hough Transform, 
2. A 2D representation of the Hough Space(s), 
3. The result images showing final detections using bounding boxes (in green) and the ground truth (in red).

b)	EVALUATION: Evaluate your detector on all of the example images. 
1. Provide the TPR and F1-score for each of the test images and their average across the 16 images in a table. Note in extra table columns the difference w.r.t. to the detection performances using only the Viola-Jones detector (Task1). 
2. Briefly note in bullet points the key merits and shortcomings of your now enhanced implementation.

c)	DETECTION PIPELINE: 
1. In a flow diagram, depict how you have combined evidence from the Hough Transform(s) and Viola-Jones detector. 
2. In bullet points, explain briefly your rationale behind the way you have combined evidence.  

## :red_circle: Subtask 3: Improving your Detector
_(10 marks)_

The final subtask allows you to develop the dartboard detector further into a direction you choose. We ask you to identify and utilise some image aspects/features able to improve detection results further. This will generally include identifying, researching, understanding and using in OpenCV one other appropriate vision approach to further improve the detection efficacy and/or efficiency

### What to say in your report (roughly 0.5-1 page):

a)	IDEA: In bullet points, explain briefly your rationale behind selecting the approach you have taken.

b)	VISUALISE: Visualise important aspects of your technique in two of the given example images of dartboards selected to best exhibit the merit of your approach. 

c)	EVALUATE: Evaluate your final detector on all of the example images, show the improvements in TPR and F1-score compared to previous approaches. Briefly note in bullet points the key merits and shortcomings of your final implementation.

## :red_circle: Notes on Your Submission for Part 1

Include your source code and your maximum 3-page PDF report, and if needed, include a readme.txt to explain how to compile/build. Your final detector program should take one parameter, that is the input image filename, and produce at least an image `detected.jpg`, which highlights detected dartboards by bounding boxes. You can use your own machines or lab machines to develop your program, but please test that it runs on the lab machines seamlessly when it comes to marking. Make sure you regularly save/backup your work and monitor your workload throughout the duration of the project
