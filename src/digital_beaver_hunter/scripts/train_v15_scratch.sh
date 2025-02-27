# Yolov8 dataset version 15
#NAME=beaver-finder-vhr-imagery.v15i.yolov5pytorch
#DATA_YAML=data_beavers/Yolo_datasets/beaver-finder-vhr-imagery.v15i.yolov5pytorch/data.yaml
#MODEL=yolov10x
# run training
#echo $DATA_YAML
yolo task=detect model=yolov10x epochs=200 data=data_beavers/Yolo_datasets/beaver-finder-vhr-imagery.v15i.yolov5pytorch/data.yaml project=Beavers name=yolov10x_datasetv15_scratch lr0=0.005 momentum=0.90 imgsz=1024 flipud=0.5 fliplr=0.5 device=\'2,3\'