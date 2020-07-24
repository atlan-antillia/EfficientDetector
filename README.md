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

At first, you have to install Microsoft Visual Studio 2019 Community Edition.<br>
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
To see the labels (id, classname, score) for the detected objects, we have updated <a href="./visualize/vis_utils.py">'automl/efficientdet/visualize/vis_utils.py</a><br>
By using the updated visualizing function <i>draw_bounding_box_on_image</i>, you can see detailed information on the detected objects as shown below:<br><br>
 
<img src="./ref-images/id_classname_score_list.png" width="80%">

<br>
<h3>
1.2 How about <a href="https://github.com/AlexeyAB/darknet">yolov4</a>?
</h3>
The following is a detected image by yolov4. All labels to the detected objects are written into console, not on the input image.<br><br>
>python darknet data/img.png<br><br>
<br>
<img src="./detected/img_by_yolov4.png" >
<br><br>
Why yolov4 doesn't draw lables on input image? It is mainly because the darkent.py of yolov4 uses draw object of skimage to draw the input image,
however the draw object has no method to draw a text string. <br>
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
<br>
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
 We would like to save the detected objects information as a text file in the following format.<br>
<pre> 
 1 car:  90%
 2 car:  87%
 3 car:  85%
 4 car:  84%
 5 car:  84%
 6 car:  83%
 7 car:  81%
 8 car:  81%
 9 person:  79%
 10 person:  77%
 11 person:  77%
</pre>

We have updated <i>inference</i> method of <a href="./inference.py">InferenceDriver</a> class in <i>automl/efficientdet/inference.py</i>
to save detected objects information as a text file.<br>
 
Furthermore, we have updated some visualization function such as a <a href="./visualize/vis_utils.py">visualize_boxes_and_labels_on_image_array</a>
in <i>automl/efficientdet/visualize/vis_utils.py</i>
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

<pre>
>python EfficientDetector.py images\img.png detected "[person,car]"
</pre>

You can see the detected image and objects information detected by above command as shown below.<br>
<br>
<img src="./detected/person-car-img.png">
<br><

<img src="./detected/person-car-img.png.txt.png">


</body>

</html>