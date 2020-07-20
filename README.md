<html>
<body>
<h1>EfficientDetector</h1>
<font size=3><b>
This is a simple python class EfficientDetector based on Brain AutoML efficientdet.<br>
</b></font>
<br>
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
pip install pycocotools<br>
</font>
</td></tr>


</table>
<br>
You have to run the following script to download an EfficientDet model chekcpoint file:<br><br>
>python DownloadCkpt.py
<br><br>
,and a sample image file:<br><br>
>python DownloadImage.py
<br><br>


Run EfficientDetector.py to detect objects in an image in the following way.<br><br>
>python EfficientDetector.py .\images\im.png
<br><br>
<img src="./detected/img.png">
<br>
Unfortunately, in this way the detected image will contain a lot of labels to the detected objects, and the original input image wll be covered by them.  <br>
If possible, the labels should be shown in somewhere like a listbox, not to be drawn on the input image as shown below <br><br>
<img src="./ref-images/html_generator.png" width="80%"><br>

See also: <a href="http://www.antillia.com/sol4py/samples/keras-yolo3/DetectedObjectHTMLFileGenerator.html">DetectedObjectHTMLFileGenerator</a>
<br>
<br>
<b>How about <a href="https://github.com/AlexeyAB/darknet">yolov4</a>?</b> <br>
The following is a detected image by yolov4. All labels to the detected objects are written into console, not on the input image.<br><br>
>python darknet data/img.png<br><br>
<br>
<img src="./detected/img_by_yolov4.png" >
<br><br>
Why yolov4 doesn't draw lables on input image? It is mainly because the darkent.py of yolov4 uses draw object of skimage to draw the input image,
however the draw object has no method to draw a text string. <br>
<br>

>python EfficientDetector.py .\images\ShinJuku.png
<br><br>
<img src="./detected/ShinJuku.jpg" >
<br>
<br>
>python EfficientDetector.py .\images\ShinJuku2.png
<br><br>
<img src="./detected/ShinJuku2.jpg" >
<br>
<br>
>python EfficientDetector.py .\images\Takashimaya2.png
<br><br>
<img src="./detected/Takashimaya2.jpg">
<br><br>
<b>
You can specify input_image_dir and output_image_dir in the following way.
</b>
<br>
>python EfficientDetector.py input_image_dir output_image_dir
<br><br>

</body>

</html>