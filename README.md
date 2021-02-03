<html>
<body>
<h1>EfficientDetector</h1>
<font size=3><b>
This is a simple python class EfficientDetector based on <a href="https://github.com/google/automl">Brain AutoML efficientdet</a>.<br>
We have changed the orginal efficientdet source code to apply filters to be able to select some specified objects only.<br>
</b></font>

<br>
<h2>1 EfficientDet Inference</h2>
<h3>
1.1 Installing Google Brain AutoML
</h3>
<font size=2>
We have downloaded <a href="https://github.com/google/automl">Google Brain AutoML</a>,  
which is a repository contains a list of AutoML related models and libraries,
and built an inference-environment to detect objects in an image by using a EfficientDet model "efficientdet-d0".<br><br>

At first, you have to install Microsoft Visual Studio 2019 Community Edition for Windows10.<br>
How to setting up an environment for AutoML on Windows 10.<br>

<table style="border: 1px solid red;">
<tr><td>
<font size=2>
pip install tensorflow==2.2.0<br>
pip install cython<br>
git clone https://github.com/google/automl<br>
cd automl<br>
git clone https://github.com/cocodataset/cocoapi<br>
cd cocoapi/PythonAPI<br>
<br> 
# Probably you have to modify extra_compiler_args in setup.py in the following way:<br>
# setup.py<br>
        #extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],<br>
        extra_compile_args=['-std=c99'],<br><br>
python setup.py build_ext install<br>
pip install pyyaml<br>
</font>
</td></tr>

</table>

<br>
Please clone <a href="https://github.com/atlan-antillia/EfficientDetector.git">EfficientDetector.git</a> in a working folder.
<pre>
>git clone  https://github.com/atlan-antillia/EfficientDetector.git
</pre>
Copy the files in that folder to <i>somewhere_cloned_folder/automl/efficinetdet/</i> folder.
<br>

You have to run the following script to download an EfficientDet model chekcpoint file:<br>
<pre>
>python <a href="./DownloadCkpt.py">DownloadCkpt.py</a>
</pre>
,and a sample image file:<br>
<pre>
>python <a href="./DownloadImage.py">DownloadImage.py</a>
</pre>

Run EfficientDetector.py to detect objects in an image in the following way.<br>
<pre>
>python <a href="./EfficientDetector.py">EfficientDetector.py</a> .\images\im.png detected
</pre>
<img src="./detected/img.png">
<br>

Unfortunately, the detected image will contain a lot of labels (classname, score) to the detected objects, 
and the original input image wll be covered by them.  <br>
If possible, the labels should be shown in somewhere like a listbox, not to be drawn on the input image as shown below <br><br>
<img src="./ref-images/html_generator.png" width="80%"><br>

See also: <a href="http://www.antillia.com/sol4py/samples/keras-yolo3/DetectedObjectHTMLFileGenerator.html">DetectedObjectHTMLFileGenerator</a>
<br>
<br>
To show the labels (id, classname, score) for the detected objects on the console window, 
we have created a new <a href="./vis_utils2.py">'./vis_utils2.py</a>
 from the original <i>automl/efficientdet/visualize/vis_utils.py.</i><br>
By using the updated visualizing function <i>draw_bounding_box_on_image_with_filters</i> in vis_utils2.py, 
you can see detailed information on the detected objects as shown below:<br><br>
 
<img src="./ref-images/id_classname_score_list.png" width="80%">

<br>
<h3>
1.2 How about <a href="https://github.com/AlexeyAB/darknet">darknet YOLOv4</a>?
</h3>
The following is a detected image by <a href="https://github.com/atlan-antillia/YOLOv4ObjectDetector">YOLOv4ObjectDetector</a>.
<br>
>python YOLOv4ObjectDetector.py images/img.png coco_detect.config
<br>
<br>
<img src="https://github.com/atlan-antillia/YOLOv4ObjectDetector/blob/main/dataset/coco/outputs/img.png" >
<br><br>
<br>

<h3>
1.3 Some inference examples of EfficientDet
</h3>
<pre>
>python EfficientDetector.py .\images\ShinJuku.png
</pre>

<img src="./detected/ShinJuku.jpg" >
<br>
<pre>
>python EfficientDetector.py .\images\ShinJuku2.png
</pre>
<img src="./detected/ShinJuku2.jpg" >
<br>
<pre>
>python EfficientDetector.py .\images\Takashimaya2.png
</pre><br>
<img src="./detected/Takashimaya2.jpg">
<br><br>
<b>
You can specify input_image_dir and output_image_dir in the following way.
</b>
<pre>
>python EfficientDetector.py input_image_dir output_image_dir
</pre>
EfficientDetector reads all jpg and png image files in the <i>input_image_dir</i>, detects objects in those images  
and saves detected images in the <i>output_image_dir</i>.
<h2>
2 Customizing a visualization process in EfficientDet
</h2>
<h3>
2.1 How to save the detected objects information?
</h3>
 We would like to save the detected objects information as a csv file in the following format.<br>
<pre>
id, class,      score, x,   y, width, height
--------------------------------------------
1,  car,         90%, 1305, 612, 326, 281
2,  car,         87%, 1245, 896, 336, 183
3,  car,         85%, 1569, 643, 328, 261
4,  car,         84%, 211,  631, 261, 203
5,  car,         84%, 1024, 376,  91,  93
6,  car,         83%, 1173, 481, 140, 127
7,  motorcycle,  81%, 1230, 747,  89, 122
8,  car,         81%, 486,  871, 293, 205
9,  car,         81%, 1107, 327,  69,  54
10,  person,     79%, 1094, 472,  40,  72
11,  motorcycle, 78%, 1082, 717,  73, 125
12,  person,     77%, 813,  593,  54, 108
</pre>
Furthermore, the number of objects in each class (objects_stats) on the detected objects as csv below.<br>
<pre>
id class     count
-------------------
1, car,        24
2, motorcycle, 16
3, person,     52
4, bus,         2
</pre>
We have updated <i>inference</i> method of <a href="./inference2.py">InferenceDriver2</a> class which was derived from 
the original <i>InferenceDriver</i>class of <i>automl/efficientdet/inference.py</i>
to save detected objects information as a text file.<br>
 
Furthermore, we have created a new file <i>vis_utils2.py</i> to define some visualization functions such as a <a href="./vis_utils2.py">visualize_boxes_and_labels_on_image_array_with_filters</a>.
<br>

<h2>
2.2 How to apply filters to detected objects?
</h2>

Imagine to select some specific objects only by specifying object-classes from the detected objects.<br>
To specify classes to select, we use the list format like this.
<pre>
  [class1, class2,.. classN]
</pre>
For example, you can run the following command to select objects of <i>person</i> and <i>car</i> from <i>images\img.png.</i><br>
<br>
Example 1: filters=[person,car]<br>

<pre>
>python EfficientDetector.py images\img.png detected "[person,car]"
</pre>
In this case, the detected image, objects, objects_stats filenames will become as shown below, with filters name. 
<pre>
person_car_img.png
person_car_img.png.csv
person_car_img.png_stats.csv

</pre>

You can see the detected image and objects information detected by above command as shown below.<br>
<br>
<br>
person_car_img.png<br>
<img src="./detected/person_car_img.png">
<br><br>

person_car_img.png.csv<br>
<img src="./detected/person_car_img.png.csv.png">
<br>
<br>
person_car_img.png_stats.csv<br>

<img src="./detected/person_car_img.png_stats.csv.png">
<br>
<br>
Example 2: filters=[person]<br>
<pre>
>python EfficientDetector.py images\ShinJuku2.jpg detected "[person]"
</pre>
<br>

person_ShinJuku2.jpg<br>
<img src="./detected/person_ShinJuku2.jpg">
<br>
<br>
person_ShinJuku2.jpg.csv<br>

<img src="./detected/person_ShinJuku2.jpg.csv.png">
<br>
<br>
person_ShinJuku2.jpg_stats.csv<br>

<img src="./detected/person_ShinJuku2.jpg_stats.csv.png">
<br>


</body>

</html>