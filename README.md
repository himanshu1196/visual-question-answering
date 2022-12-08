# visual-question-answering
https://pantelis.github.io/artificial-intelligence/aiml-common/projects/vqa/index.html
https://arxiv.org/pdf/1706.01427.pdf


Create conda environment from environment.yml file

Usage :

Create environment :
$ conda env create -f environment.yml

Activate environment :
$ conda activate RN3

Generate sort-of-clevr dataset :
$ conda install pandas
$ python sort_of_clevr_generator.py

Train the binary RN model : 
$ python main.py --model Bigger_RN

Train the State RN model:
$ python main.py --model State_RN


