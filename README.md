# Detectron2-Deepstream
A sample of how to use detectron2 with deepstream. 
I have implemented the deepstream with both mask_rcnn and faster_rcnn.

## Reference:
I have taken reference from the official deepstream documentation.
Especially the objectDetector_Yolo.

## System Environment:
Jetson AGX ORIN (Ubuntu 20.04 focal)
Jetpack - 5.1.1 [L4t 35.3.1]
CUDA: 11.4.315
cuDNN : 8.6.0.166
TensorRT: 8.5.2.2
Deepstream: 6.2

## Prerequisites:
1. Please follow the [TensorRT's](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples/python/detectron2) github repo on how to build onnx for mask_rcnn_R_50_FPN_3x model.

2. Please build the tensorrt engine using the ./trtexec tool since my deepstream example cannot build the engine yet.

3. Copy both your onnx file and engine file into this directory after you have git clones this repo. Please be sure to change the name of onnx and engine files in the config_infer_primary_detectron2.txt.

## Implement Deepstream:
```
cd ~/
git clone https://github.com/RajUpadhyay/Detectron2-Deepstream.git
```
```
cd Detectron2-Deepstream/
make -C nvdsinfer_custom_impl_detectron2_seg/
```

Run using the deepstream-config or gst-pipeline
```
deepstream-app -c deepstream_app_config.txt
```
```
GST_DEBUG=3 gst-launch-1.0 uridecodebin3 uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_qHD.mp4 ! nvstreammux0.sink_0 nvstreammux name=nvstreammux0 batch-size=1 batched-push-timeout=40000 live-source=False width=1920 height=960 ! nvinfer config-file-path=config_infer_primary_detectron2.txt ! nvvidconv ! nvdsosd ! nvvidconv ! fpsdisplaysink sync=0
```

## Note:
Even if some of you are not able to implement it due to version error or something, important thing is the custom parser function needed by deepstream to implement your detectron2 model if you already have a tensorrt engine, so please refer it.

Also if there are any mistakes or any suggestions, let me know.