#Downy Mildew Disease detection
TensorFlow object detection + Downy Mildew Disease detection

1. Install labelImg: pip3 install labelImg
2. Directory structure: Object-Detection->data
					->training
					->images
						->train
						->test

3.change code xml_to_csv.py
def main():
    for directory in ['train','test']:
        image_path = os.path.join(os.getcwd(), 'images/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)
        print('Successfully converted xml to csv.')

4. Run python3 xml_to_csv.py

5. Change code in  generate_tfrecord.py
# Add all the classes

def class_text_to_int(row_label):
    if row_label == 'downy':
        return 1
    else:
        return 0
    
6. Run 
python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/

python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/

Now, in your data directory, you should have train.record and test.record.

7. Clone fresh tensorflow models
git clone https://github.com/tensorflow/models.git

Then, follow the installation instructions:

sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install jupyter
sudo pip install matplotlib

8. Go to research foilder and open terminal
# From tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.

9.set python path
# From tensorflow/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

10. Install in to python 
sudo python3 setup.py install

11. wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

Get Mobilnet configuration files


12. Get the sample config files from models/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config
	edit the config file with all the path
	#change num of classes
	num_classes: 1 
	#change the batch size
	batch_size: 24
	#change if you want
	initial_learning_rate: 0.004

	fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"

	input_path: "data/train.record"

	label_map_path: "data/object-detection.pbtxt"

	input_path: "data/test.record"

	label_map_path: "training/object-detection.pbtxt"

	After Editing above items, Put the config in the training directory.

13. Inside training dir, add object-detection.pbtxt: Add all your classes.

item {
  id: 1
  name: 'downy'
}

14. Add Object-Detection.pbtxt file in into data directory

15. Copy data, image, training, ssd_mobilenet_v1_coco_11_06_2017 (unzipped) in to object-detection directory)
merge the data directory with existing directory

16. Run the command: python3 model_main.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

17. If any error wrt 'nets' then set the python Path again on research directory # From tensorflow/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

18. find the temp directory where the event files are generated and run 
#Try-->

tensorboard --logdir='/tmp/tmp8ecnodas'


in browser open--> http://127.0.0.1:6006/


19. run
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-23361 \
    --output_directory invoice_inference_graph_23361

20. Run--> jupyter notebook

21. Goto object_detection_tutorial.ipynb directry
-->In Variables 
# What model to download.
MODEL_NAME = 'invoice_graph_2'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES=1

-->delete Download Model content(optional)	

-->In Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 10) ]#range() as per the number of images
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

22. goto Cell
	 -->Run All
Take a look at test results <reuslt>
