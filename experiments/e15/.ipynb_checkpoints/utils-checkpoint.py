import torch
from torch.utils.data import DataLoader
from miditok import REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator
from pathlib import Path
import json
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import balanced_accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F

seed = 42
if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define a custom DatasetMIDI subclass
class CustomDatasetMIDI(DatasetMIDI):
    def __init__(self, files_paths, labels, tokenizer, max_seq_len, bos_token_id, eos_token_id):
        super().__init__(files_paths, tokenizer, max_seq_len, bos_token_id, eos_token_id)
        self.labels = labels

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["labels"] = self.labels[idx]  # Add labels to the item dictionary
        return item

def create_data_loader(scores_df,paths_column_name):
  
  midi_paths = []
  labels=[]
  paths=scores_df[paths_column_name]
  paths=paths.apply(lambda x: x.replace('../author-profiling-in-symbolic-music/','../../train data/'))

  for i,score in enumerate(paths):

      midi_paths.append(Path(score))

      integer_label= 0 if scores_df['composer_gender'][i] == 'Male' else 1

      labels.append(torch.tensor(integer_label))  # Modify this line to extract the label

  # Initialize the tokenizer
  tokenizer = REMI.from_pretrained("Natooz/Maestro-REMI-bpe20k")


  # Initialize the dataset
  dataset = CustomDatasetMIDI(
      files_paths=midi_paths,
      labels=labels,  # Pass the labels to the dataset
      tokenizer=tokenizer,
      max_seq_len=2045,
      bos_token_id=tokenizer.pad_token_id,
      eos_token_id=tokenizer["BOS_None"]
  )

  # Initialize the collator
  collator = DataCollator(tokenizer.pad_token_id)

  data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=len(scores_df))

  return data_loader


def get_feature_vectors(dataloader, dataframe, set_type, feature_tensors):

    from transformers import AutoModelForCausalLM
    from tqdm import tqdm 
    
    for batch in dataloader:
        print()
        
    torch.set_default_device("cpu")
    # Load model
    
    ##======================INSTANTIATE MODEL==========================##
    model = AutoModelForCausalLM.from_pretrained(
        "Natooz/Maestro-REMI-bpe20k",
        trust_remote_code=True,
        torch_dtype="auto",
        output_hidden_states=True
    )
    
    ##=========GET FEATURE VECTORS FROM LAST HIDDEN LAYER===============##
    if feature_tensors == False:    
    #Ensure model is in evaluation mode
        model.eval()
        last_hidden_state_list=[]
        
        for i in tqdm(range(0,batch['input_ids'].shape[0],5), desc="computing feature tensors"):
            
            #Get model outputs including hidden states
            with torch.no_grad():
                outputs = model(batch['input_ids'][i:i+5])
            
            # Extract hidden states (output hidden states are a tuple, each element corresponds to different layers)
            hidden_states = outputs.hidden_states
            last_hidden_state=hidden_states[-1]
            last_hidden_state_list.append(last_hidden_state)
            
        loader_tensor_list=last_hidden_state_list

    elif feature_tensors == True:
        loader_tensor_list=torch.load(f'tensor_list_{set_type}.pt')
    
    
    ##======================FLATTEN FEATURE VECTORS FOR MLP==========================##
    flattened_tensor_list=[]
    
    for batch in range(0,len(loader_tensor_list)):
    
        for element in range(0,len(loader_tensor_list[batch])):
            
            flat_tensor=torch.flatten(loader_tensor_list[batch][element])
            flattened_tensor_list.append(flat_tensor)

    numpy_arrays=[item.numpy() for item in flattened_tensor_list]
    
    features_df=pd.DataFrame(data={'feature_vectors':numpy_arrays})
    
    features_df = features_df['feature_vectors'].apply(pd.Series)  
        
    composer_gender=dataframe['composer_gender'].apply(lambda x: 0 if x=='Male' else 1)

    features_df['label']=composer_gender

    return features_df


##===========MLP CLASSIFIER====================##
    
class DatasetMLP(Dataset):

    def __init__(self,data):
        self.data=data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,ind):
        x=self.data[ind][:-1]
        y=self.data[ind][-1]

        return x,y

class TestDataset(DatasetMLP):
    def __getitem__(self,ind):
        x=self.data[ind]
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, output_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out

from sklearn.metrics import balanced_accuracy_score

def evaluate(model, dataloader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set model to evaluation mode
    predictions = []
    true_labels = []
    probabilities_list=[]
    losses = []

    with torch.no_grad():
        for input_data in dataloader:
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device).long()

            output = model(x)
            _, predicted = torch.max(output, 1)
            probabilities = F.softmax(output, dim=1)

            batch_predictions = predicted.cpu().detach().numpy().tolist()
            batch_true_labels = y.cpu().detach().numpy().tolist()
            batch_predicted_probas = probabilities.cpu().detach().numpy().tolist()

            predictions.extend(batch_predictions)
            true_labels.extend(batch_true_labels)
            probabilities_list.extend(batch_predicted_probas)

            # Compute loss
            loss = criterion(output, y)
            losses.append(loss.item())

    # Compute average validation loss
    avg_loss = sum(losses) / len(losses)

    # Compute balanced accuracy
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, predictions)

    return balanced_accuracy, avg_loss, probabilities_list,predictions, true_labels




##=== declare function here

def train_MLP_classfier(train_feature_vectors,val_feature_vectors,k):

    train_set_mlp=DatasetMLP(np.array(train_feature_vectors))
    val_set_mlp=DatasetMLP(np.array(val_feature_vectors))
    
    batch_size=20
    
    train_dataloder_mlp=DataLoader(train_set_mlp,
                               batch_size=batch_size,
                               shuffle=True)  
    
    val_dataloder_mlp=DataLoader(val_set_mlp,
                               batch_size=batch_size,
                               shuffle=False)  
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dim = 523264
    model = MLP(input_dim).to(device)
    
    initial_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    print(model)
    
    epochs = 10
    
    model.train()
    train_avg_loss_list=[]
    val_avg_loss_list=[]
    train_balanced_accuracy_list=[]
    val_balanced_accuracy_list=[]
    
    for epoch in range(epochs):
        losses = []
        predictions = []
        true_labels = []
        for batch_num, input_data in enumerate(train_dataloder_mlp):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device).long()  # Ensure y is of type long for CrossEntropyLoss
        
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())
        
            optimizer.step()
        
            # Convert predictions to class labels (0 or 1)
            _, predicted = torch.max(output, 1)
            # Apply softmax to get probabilities
            
            batch_predictions = predicted.cpu().detach().numpy().tolist()
            batch_true_labels = y.cpu().detach().numpy().tolist()
        
            predictions.extend(batch_predictions)            
            true_labels.extend(batch_true_labels)
        
            #if batch_num % 40 == 0:
            #    print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
        # Step the scheduler
        scheduler.step()
        
        train_balanced_accuracy = balanced_accuracy_score(true_labels, predictions)
        train_avg_loss=sum(losses)/len(losses) 
        print('Epoch %d | Train Loss %6.2f| Train Balanced Accuracy %6.2f' % (epoch, train_avg_loss    ,train_balanced_accuracy))
        
        val_balanced_accuracy, val_avg_loss, probabilities_list, val_predictions, val_true_labels = evaluate(model, val_dataloder_mlp,criterion)
        print('Epoch %d | Validation Loss %6.2f| Validation Balanced Accuracy: %6.2f' % (epoch, val_avg_loss, val_balanced_accuracy))
        
        train_avg_loss_list.append(train_avg_loss)
        val_avg_loss_list.append(val_avg_loss)
        
        train_balanced_accuracy_list.append(train_balanced_accuracy)
        val_balanced_accuracy_list.append(val_balanced_accuracy)
        
        
        # Convert predictions and true labels to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)      

    predictions_df_train_e17=pd.DataFrame(data={'labels':true_labels,'predictions':predictions})
    #predictions_df_train_e17.to_csv(f'predictions_df_train_e17_k{k}.csv')
    
    metrics_df=pd.DataFrame(data={'train_avg_loss':train_avg_loss_list,
                                'train_balanced_accuracy':train_balanced_accuracy_list,
                                'val_avg_loss':val_avg_loss_list,
                                'val_balanced_accuracy':val_balanced_accuracy_list})
    
    #metrics_df.to_csv(f'metrics_df_e17_k{k}.csv',index=False)

    predictions_df_val_e17=pd.DataFrame(data={'labels':val_true_labels,
                                              'probabilities':probabilities_list,
                                             'predictions':val_predictions})
    #predictions_df_val_e17.to_csv(f'predictions_df_test_e17_k{k}.csv')
    #print('exported files')
    
    return predictions_df_train_e17,metrics_df,predictions_df_val_e17
    



        
        
        





















    