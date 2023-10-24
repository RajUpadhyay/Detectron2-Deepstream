## Prerequisites:
1. Please follow the [TensorRT's](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples/python/detectron2) github repo on how to build onnx for mask_rcnn_R_50_FPN_3x model and then edit it for faster_rcnn model. Its very similar so it should be easy.

2. Please build the tensorrt engine using the ./trtexec tool since my deepstream example cannot build the engine yet.

3. Copy both your onnx file and engine file into this directory after you have git clones this repo. Please be sure to change the name of onnx and engine files in the config_infer_primary_detectron2.txt.

## Implement Deepstream:
```
cd Detectron2-Deepstream/
git checkout frcnn
make -C custom_bbox_parser/
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