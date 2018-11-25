# traffic-monitor-lab

This repository is intended to keep track of what I'm working on and to be used also
as a sandbox for testing new features, technologies to be incorporated to the smart-traffic-sensor
software.

# Steps to re-traing your own network by using tensorflow and transfer-learning techniques

1. Download tensorflow models

```bash
     git clone https://github.com/tensorflow/models
```

2. Prepare your training data

For this you can reuse some already existing database or generate your own data. To train
the vehicles-detection module of traffic monitor I generated my own database of vehicles
following the PASCAL VOC format. This is a well-known format that can be used later to generate
the traing.record and testing.record TFRecord files needed by tensorflow to re-train an already
existing network.

To generate the TFRecords a modified version of create_pascal_tf_record.py script will be used. Basically,
the original script provided by tensorflow is ready to receive as input a dataset in PASCAL VOC format directory
structure and use it to generate the TF records. However, the PASCAL structure is too complicated
for what's needed, so a modified script will be used so it can handle the following directory structure:

```
 dataset \
         \-- annotations
         \-- images
         \-- files.txt
         \-- labels.pbtxt
```

Where:
* annotations   : directory containing the XML annotations
* images        : directory containing the JPEG images
* files.txt     : a file listing the images to be used for training
* labels.pbtxt  : file containg the classes labels

The file **labels.pbtxt** must contain a list of the object classes, i.e:

```
item {
  id: 1
  name: "car"
}
item {
  id: 2
  name: "motorcycle"
}
...
```

Once the data is ready you can create the **test.record** and **traing.cord** as following:

```
   python machine-learning/tools/create_pascal_tf_record.py --data_dir vehicles-dataset/ --output_path data/test.record --files test-files-balanced.txt 
```

3. Re-training the network

  Before starting the re-training script, you have to decide which network you will be using as starting point.
  There are several pre-trained networks in [tensorflow zoo model][1] repository, just pick one that fullfill your
  needs. Since in my case I'm looking for a fast object detection network, my first choice is SSD mobilenet V2
  network. This network provides a good tradeoff between accurracy and speed. Once you decice which network to
  use, you have to download its **.pb** file and configuration file. In the case of SSD mobilenet V2 you can get
  the configuration file from the following [link][2]. This file contains severaltraining parametres needed
  by this network, but you just need to fine-tune some of them, for instance:

   * **num_classes**: as its name indicate, this parameter indicates the number of classes
   * **input_path** (tf_record_input_reader record): this parameter must point to the train.record file generated previously
   * **label_map_path** (tf_record_input_reader record): this parameter must point to the labels.pbtxt file generated previously

  For example in my setup, the configuration file looks like:

```
train_input_reader: {
  tf_record_input_reader {
    input_path: "data/train.record"
  }
  label_map_path: "data/labels.pbtxt"
}
```

   The same must be done for the testing records, pointing to the right files. At this point you have all what you need to start the
   re-training script. My data directory structure is as following:

```
 data \
      \-- labels.pbtxt
      \-- ssd_mobilenet_v2_coco.config
      \-- test.record
      \-- train.record
```

   To launch the re-training you have to:

   1. Go to the directory models/research/object_detection/legacy
   2. Copy the data directory to hold the files needed for training (see above)
   3. Copy the images directory to the local directory
   4. In models/research, you have to execute: protoc object_detection/protos/*.proto --python_out=.
   5. Launch the training script as following:

```bash
   export PYTHONPATH=../..:../../slim/
   python train.py --logtostderr --train_dir training/ --pipeline_config_path=./data/ssd_mobilenet_v2_coco.config
```

   This will start the training process that may last for a while. During the same you can monitor the loss-function
   so you can stop it whenever it reaches a reasonable value. By default it will stop automatically after 200k steps.
   The training script from time to time saves a checkpoint (you can see them at training_dir/checkpoint).


3. Generate your model

   At this point you have already ran the training script and generated our model, but the results are still in the intermediate
   format. In order to use the generated model outside you have to freeze it to a **.pb** network graph file. For this you have to
   use the export_inference_graph.py script (from legacy directory):

```bash
python ../export_inference_graph.py --input_type image_tensor --pipeline_config_path data/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-XXX --output_directory mymodel/
```

  where XXX are digits corresponding to the checkpoint to be used.

  Finally, this generates a new directory **mymodl** where the **.pb** plus other model data are saved.


[1]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[2]: https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config
