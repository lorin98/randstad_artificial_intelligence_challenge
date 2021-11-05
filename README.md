# Randstad Artificial Intelligence Challenge

This repository contains the proposed solution for the [Artificial Intelligence Challenge (NLP)](https://www.vgen.it/randstad-artificial-intelligence-challenge/) powered by [Randstad Italia](https://www.randstad.it/) and [VGen](https://www.vgen.it).

This national competition consists in a multi-class classification task, where a free-chosen machine learning algorithm aims to correctly associate job offer instances to the most likely job label (please, refer to the official website above for further details on the challenge objectives).

The proposed solution exploits the [PyTorch](https://pytorch.org/) framework and, in particular, the [PyTorch Lightning](https://www.pytorchlightning.ai/) library. The model is the italian adaptation of the [BERT](https://arxiv.org/abs/1810.04805) transformer from [HuggingFace](https://huggingface.co/) to extract contextualized information about sentences (i.e., the job offers).

**[EDIT]** The project has been ranked at the 4th place and marked as one of the best performing approaches by registering the highest score on a par with other four candidates (complete results available at this [link](https://www.vgen.it/risultati-randstad-artificial-intelligence-challenge/)). The certificate of the challenge completion is attached [here](./Randstad_Artificial_Intelligence_Challenge_certificate.pdf).

## Run the project

We provide a commented Colab/Jupyter [notebook](./Randstad_Artificial_Intelligence_Challenge.ipynb) to run the project. In addition, you are free to execute the associated [script](./run_model.py), which is a copy of the notebook, as follows:
```
pip install -r requirements.txt
python run_model.py
```
The script will preprocess the datasets, perform training and/or testing according to the desired settings and show the performances of the model. Please, respect the directory organization explained in the script or change the path to the files to properly execute the program.

## Author

Lorenzo Nicoletti
