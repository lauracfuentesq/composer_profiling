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
  paths=paths.apply(lambda x: x.replace('../author-profiling/','../'))

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
      max_seq_len=1024,
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
        
        
        





















    