# visual-question-answering
https://pantelis.github.io/artificial-intelligence/aiml-common/projects/vqa/index.html
https://arxiv.org/pdf/1706.01427.pdf

Refer to rnvqa.ipynb for problem statement, RN approach, data description, model details, and instructions on how to run the code in Google colab.

Create conda environment from environment.yml file

Usage :

Create environment :
$ conda env create -f environment.yml

Activate environment :
$ conda activate RN3

Generate sort-of-clevr dataset :
$ python sort_of_clevr_generator.py

Train the binary RN model : 
$ python main.py

Directory structure :
1. data/ :
    sort-of-clevr-original.pickle - contains train and test dataset in form of (image,question,answer) tuples
    train_descriptions.csv - contains state descriptions of images in train data
    test_descriptions.csv - contains state descriptions of images in test data
2. model/ :
    trained models are saved here after each training epoch by main.py
3. pretrained_models/ :
    Best performing models trained for several epochs are saved here
4. runs/
    Training and test performance metrics such as accuracy and loss for each dataset for tensorboard