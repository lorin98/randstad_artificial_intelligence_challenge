'''
To properly run this script, I suggest to install the following libraries:

scikit-learn==0.24.1
tqdm==4.59.0
torchvision==0.10.0
torch==1.9.0
transformers==4.6.1
pytorch_lightning==1.3.5
torchmetrics==0.4.1

The main library is PyTorch together with the PyTorch Lightning framework, used to define, declare and save the model.
The model is the italian adaptation of the BERT transformer from HuggingFace to extract contextualized information about sentences (i.e., the job offers). Consequently, the 'transformers' library is needed.
At the end of the training, the model weights are saved in a .ckpt file, a checkpoint created by Pytorch Lightning thanks to one specific callback provided by the library to save the best model in terms of performances.

This script requires the following directory settings:
./
|_ run_model.py
|_ model_checkpoint.ckpt
|_ Dataset_Randstad-Challenge/
                        |_ train_set.csv
                        |_ test_set.csv
                        
However, you are always free to change this organization by specifing the new path to the files in the global variables below (TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_WEIGHTS)

With no changes, this script only tests the model. If you want to re-train, please change the global variables below (DO_TRAIN, DO_TEST). I strongly suggest to train only with a strong GPU because of the huge dimension of BERT. Moreover, the testing phase is composed by a first evaluation by Pytorch Lightining (together with the dataset preprocessing) and a 'deeper' evaluation to compute also the other metrics. Please, be aware that this step may take some time with no GPU (10/15 minutes for my tests with no GPU).

Thank you for your attention,
Lorenzo Nicoletti
'''


import numpy as np
import torch
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, AdamW
import pytorch_lightning as pl
import csv
from tqdm import tqdm
from typing import Optional, List, Dict, Union, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report


# global variables to the paths of interest
TRAIN_DATA_PATH = 'Dataset_Randstad-Challenge/train_set.csv'
TEST_DATA_PATH = 'Dataset_Randstad-Challenge/test_set.csv'
# weights of the best model
MODEL_WEIGHTS = 'model_checkpoint.ckpt'

# do train and/or test
DO_TRAIN = False # we do not want to train ...
DO_TEST = True # ... but only to test our best model

# device on which we will store everything
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mapping from word labels to number labels
mapping = {'Java Developer': 0,
           'Web Developer': 1,
           'Programmer': 2,
           'System Analyst': 3,
           'Software Engineer': 4}

# to guarantee repetability of the experiments if needed
def set_seed():
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
# function to read the csv dataset and create a list where each entry is
# a dictionary with <job offer, label> pairs
def csv2list(path) -> List[Dict]:
    with open(path, encoding="utf8") as f:
        # skip header
        next(f)
        reader = csv.reader(f, delimiter=',')

        dataset = []
        for row in reader:
            # read dataset line by line
            data_item = {'job_offer': row[0]}
            data_item['label'] = row[1]

            dataset.append(data_item)
      
    return dataset

# evaluating function that computes precision, recall, F1 and accuracy of the model
# and the related classification report
def evaluate_results(labels, predictions):

    print(classification_report(labels, predictions, labels=list(mapping.keys()), digits=4))
    
    p = precision_score(labels, predictions, average='macro')
    r = recall_score(labels, predictions, average='macro')
    a = accuracy_score(labels, predictions)
    f = f1_score(labels, predictions, average='macro')
    
    print(f'# precision: {p:.4f}')
    print(f'# recall: {r:.4f}')
    print(f'# acc: {a:.4f}')
    print(f'# f1: {f:.4f}')
    
       
############################# DATASET AND DATALOADER #############################

# dataset class
class RandstadDataset(Dataset):

    def __init__(self,
                data: list,
                device: str):
      
        self.samples = []
        self.data = data
        
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-italian-cased',
                                                      do_lower_case=True)

        self.init_data()

    def init_data(self):

        for elem in self.data:
            # mapping the label
            label = torch.tensor(mapping[elem['label']])

            # the tokenizer derives the tokens ...
            tokens = self.tokenizer.tokenize(elem['job_offer'])

            # ... and encodes them to get the input_ids, attention_mask and token_type_ids
            encoding = self.tokenizer.encode_plus(tokens,
                                                  max_length=512,
                                                  truncation=True,
                                                  add_special_tokens=True,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=True,
                                                  return_tensors='pt')
            
            self.samples.append([encoding['input_ids'].to(self.device),
                                encoding['attention_mask'].to(self.device),
                                encoding['token_type_ids'].to(self.device),
                                label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    

# collate function to be used in the dataloaders
# a batch must be composed by elements of the same size -> this collate function
# realizes this goal by padding to the maximum size of the longest element in each batch
def collate_fn(data_elements: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    input_ids = [de[0].squeeze(0) for de in data_elements]
    attention_masks = [de[1].squeeze(0) for de in data_elements]
    token_type_ids = [de[2].squeeze(0) for de in data_elements]
    labels = [de[3] for de in data_elements]
    
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)

    return input_ids, attention_masks, token_type_ids, torch.tensor(labels).to(DEVICE)


# Lightning Data Module from Pytorch Lightning to store all the data I need
class RandstadDataModule(pl.LightningDataModule):

    def __init__(self,
                train_data_path: str,
                test_data_path: str,
                batch_size: int):
      
        super().__init__()

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        
        self.batch_size = batch_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # setup function to instantiate the datasets
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = RandstadDataset(self.train_data_path, DEVICE)
            self.val_dataset = RandstadDataset(self.test_data_path, DEVICE)
        elif stage == 'test':
            self.test_dataset = RandstadDataset(self.test_data_path, DEVICE)

    # create train dataloader
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn)

    # create test dataloader
    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn)


##################################################################################


############################# MODEL DEFINITION SECTION #############################

# BERT pre-trained model (that will be put inside the Lightning Module)
class MyBERT(BertPreTrainedModel):
  
    def __init__(self, config):
        super().__init__(config)

        # we basically have BERT + dropout + Linear layer for classification
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # the linear layer has (768=hidden size of BERT x 5=num classes) dimension
        self.output = torch.nn.Linear(config.hidden_size, 5)

        self.init_weights()
    
    def forward(self, input_ids, attention_mask, token_type_ids):
      
        # flow of information through the layers of the network
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        dropout_output = self.dropout(bert_output[1])
        logits = self.output(dropout_output).squeeze(-1)

        return logits

    
# Lightning Module from Pytorch Lightning
class RandstadModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-italian-cased',
                                                      do_lower_case=True)

        # transformers fine-tuning 'black magic': with one line of code we can fine-tune the model
        self.bert = MyBERT.from_pretrained('dbmdz/bert-base-italian-cased', config=config)

        # metrics to monitor and test the model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.test_f1 = torchmetrics.F1(5)

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, token_type_ids, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        logits = self.bert(input_ids, attention_mask, token_type_ids)
        # we get a probability distribution (the highest logits corresponds to the
        # most likely label)
        pred = torch.softmax(logits, dim=-1)

        result = {'logits': logits, 'pred': pred}

        # compute loss
        if labels is not None:
            loss = self.loss(logits, labels)
            result['loss'] = loss

        return result
    
    def training_step(
        self, 
        batch: Tuple[torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:

        # single training step: we take the output of the model and monitor loss and F1
        forward_output = self.forward(*batch)
        self.log('train_loss', forward_output['loss'], prog_bar=True)
        self.test_f1(forward_output['pred'], batch[3])
        self.log('train_f1', self.test_f1, prog_bar=True)
        return forward_output

    def test_step(
        self,
        batch: Tuple[torch.Tensor],
        batch_idx: int
    ):
        # single testing step: we take the output of the model and test the F1
        forward_output = self.forward(*batch)
        self.test_f1(forward_output['pred'], batch[3])
        self.log('test_f1', self.test_f1, prog_bar=True)

    def loss(self, pred, y):
        return self.loss_fn(pred, y)

    def configure_optimizers(self):
      
        # standard initialization of the AdamW optimizer according to the
        # transformers literature
        param_optimizer = list(self.bert.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=2e-5,
                eps=1e-8
                )
        return optimizer

    # in the predict function we take as input a job offer and we output the most
    # likely label according to our network
    def predict(self, sentence: str) -> str:

        # we proceed again with the tokenization and the encoding as done in the preprocessing
        tokens = self.tokenizer.tokenize(sentence)

        encoding = self.tokenizer.encode_plus(tokens,
                                              max_length=512,
                                              truncation=True,
                                              add_special_tokens=True,
                                              return_attention_mask=True,
                                              return_token_type_ids=True,
                                              return_tensors='pt')
        
        # doing inference
        with torch.no_grad():
            pred = self(encoding['input_ids'].to(self.device),
                        encoding['attention_mask'].to(self.device),
                        encoding['token_type_ids'].to(self.device))['pred']
            
        # we output only the mapped label
        # the index of max value correspond to the class
        pred = torch.max(pred, -1).indices
        for key, value in mapping.items():
            if value == pred:
                label = key
        return label 

    
####################################################################################
    
def main():
    
    print('======This is my solution for the Randstad Artificial Intelligence Challenge...======\n')
    
    set_seed()
    
    train_data = test_data = csv2list(TRAIN_DATA_PATH)
    test_data = test_data = csv2list(TEST_DATA_PATH)
    dm = RandstadDataModule(train_data, test_data, batch_size=8)
    print('======Train and test datasets loaded.======\n')
    
    # two callbacks:
    # - early stopping to avoid overfitting
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='train_loss',  # the value that will be evaluated to activate the early stopping of the model.
        patience=3,  # the number of consecutive attempts that the model has to raise (or lower depending on the metric used) to raise the "monitor" value.
        verbose=True,  # whether to log or not information in the console.
        mode='min', # wheter we want to maximize (max) or minimize the "monitor" value.
    )

    # - best checkpoint saving to reach the highest results later
    check_point_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_f1',  # the value that we want to use for model selection.
        verbose=True,  # whether to log or not information in the console.
        save_top_k=3,  # the number of checkpoints we want to store.
        mode='max',  # wheter we want to maximize (max) or minimize the "monitor" value.
        dirpath='ckpt/',  # output directory path
        filename='randstad_classifier'+'-{epoch}-{val_f1:.4f}'  # the prefix on the checkpoint values. Metrics store by the trainer can be used to dynamically change the name.
    )

    trainer = pl.Trainer(max_epochs=10,
                         gpus=1 if torch.cuda.is_available() else 0,
                         progress_bar_refresh_rate=50,
                         callbacks=[early_stopping, check_point_callback])

    
    if DO_TRAIN:
        print('======The user has requested a training...This operation may take a while.======\n')
        
        # bert configuration with 5 classes, dropout of 0.0 and 768 as defualt hidden size
        config = BertConfig.from_pretrained('dbmdz/bert-base-italian-cased',
                                            num_labels=5,
                                            hidden_dropout_prob=0.0, # experiments have shown that no dropout leads to the best results
                                            hidden_size=768)
        # model creation
        model = RandstadModel(config).to(DEVICE)
        
        # training completed handled by the Trainer of Pytorch Lightning
        trainer.fit(model=model, datamodule=dm)
        
        print('======Training completed.======\n')
    
    if DO_TEST:
        print('======Loading model...This operation may take a while.======\n')
        # load the best model
        model = RandstadModel.load_from_checkpoint('model_checkpoint.ckpt')
        print('======Model successfully loaded.======\n')
        
        print('======Starting evaluation with Pytorch Lightning.======\n')
        trainer.test(model=model, datamodule=dm)
        print('======Evaluation of Pytorch Lightning completed.======\n')

        print('======Starting a "deeper" evaluation.======\n')
        # set to eval() for inference
        model.eval()
        model.to(DEVICE)
        
        gold_labels = [elem['label'] for elem in test_data]

        # collect the predictions by analysing every single sentence
        predictions = []
        for elem in tqdm(test_data):
            predictions.append(model.predict(elem['job_offer']))

        evaluate_results(gold_labels, predictions)

        print('======Final evaluation completed...======\n')
        
        # this will overwrite my 'predictions.csv' file; please, do not execute it
        '''
        with open('predictions.csv', encoding='utf8', mode='w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            
            writer.writerow(['Job_description', 'Label_true', 'Label_pred'])
            
            for i in tqdm(range(len(test_data)), desc='Saving to csv'):
                writer.writerow([test_data[i]['job_offer'], gold_labels[i], predictions[i]])
                
        print('======Predictions stored in the csv file...======\n')
        '''
              
        print('======Challenge completed...Bye.======\n')

if __name__ == '__main__':
    main()
    
    
    
    