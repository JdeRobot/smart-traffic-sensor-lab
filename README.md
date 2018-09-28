# traffic-monitor-lab

This repository is intended to keep track of what I'm working on and to be used also
as a sandbox for testing new features, technologies to be incorporated to the traffic-monitor
software.

# Steps to re-traing your own network by using tensorflow and transfer-learning techniques

1. Download tensorflow models

```bash
     git clone https://github.com/tensorflow/models
```

2. Prepare your training data

For this you have to reuse some already existing database or generate your own data. To train
the vehicles-detection module of traffic monitor I generated my own database of vehicles
following the PASCAL VOC format. This is well-known format that can be used later to generate
the traing.record and testing.record TFRecord files needed by tensorflow to re-train an already
existing network.

To generate the TFRecords I'll be using a modified version of create_pascal_tf_record.py script. Basically,
the original script provided by tensorflow is ready to receive as input a dataset in PASCAL VOC format directory
structure and use it to generate the TF records. However, I find the PASCAL structure a little to complicated
to what I need (at least at this moment), so I modified the original script a little bit to handle the following
directory structure:

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

3. Re-training the network

  Before starting the re-training script, we have to decide which network we will be using as starting point.
  There are several pre-trained networks in [tensorflow zoo model][1] repository, just pick one that full-fill our
  needs. Since in my case I'm looking for a fast object detection network, my first choice is SSD mobilenet V2
  network. This network provides a good tradeoff between accurracy and speed. Once we decice which network to
  use we have to download its **.pb** file. At this point we have all what we need to start the re-training script.

  To launch the re-training we have to:

   1. Go to the directory models/research/object_detection/legacy
   2. Create a data directory to hold the files needed for training (labels.pbtxt, ssd_mobilenet_v2_coco.config, test.record, train.record)
   3. Copy the images directory to the local directory
   4. Launch the training script as following:

```bash
   python train.py --logtostderr --train_dir training/ --pipeline_config_path=./data/ssd_mobilenet_v2_coco.config
```

   This will start the training process that may last for a while. During the same you can monitor the loss-function
   so you can stop it whenever it reaches a reasonable value. By default it will stop automatically after 200k steps.
   The training script from time to time saves a checkpoint (you can see them at training_dir/checkpoint).


3. Generate your model

   At this point we have already ran the training script and generated our model, but the results are still in the intermediate
   format. In order to use the generated model outside we have to freeze it to a **.pb** network graph file. For this we have to
   use the export_inference_graph.py script (from legacy directory):

```bash
python ../export_inference_graph.py --input_type image_tensor --pipeline_config_path data/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-XXX --output_directory mymodel/
```

  where XXX are digits corresponding to the checkpoint to be used.

  Finally, this generates a new directory **mymodl** where the **.pb** plus other model data are saved.


[1]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
