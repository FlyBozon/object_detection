# Objects detection using NN

Hej the goal of that project is to train a nn to detect specific objects on different images. To rin it by youself firstly run `run_all.sh` by simply writing:

`chmod +x run_all.sh`

`./run_all.sh 50` <- where 50 is a number of random images for dataset u want to download


### Next part would lately be added to anouther run_train.sh script, but for now follow instructions 
#### (actually writing them for myself to not to forget XD):

Then lets move to training a nn. We would use YOLO (you only look once) for that. Firstly create a python enviroment:

`python3 -m venv yolo_env`

`source yolo_env/bin/activate`

`pip install ultralytics`


### to use TensorBoard:

`pip install tensorboard`

`tensorboard --logdir runs/detect/yolo_tensorboard`

### to add ClearML:
1. Create a ClearML account at https://app.clear.ml/
2. Install the clearml package: `pip install clearml`
3. Configure ClearML: `clearml-init`
4. Use instruction from their page, generate and paste credentials to the terminal

### NN visualization using Netron
`pip install netron`
then run `netron your_network.pt` that would open your network in browser