import plotly.express as px
import pandas as pd
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import plotly.graph_objects as go
import numpy as np





def balanced_accuracy_by_epoch(metrics_df):

    if 'val_loss' in metrics_df.columns:

        metrics_per_epoch=metrics_df[~metrics_df['val_loss'].isna()]

    elif 'val_avg_loss' in metrics_df.columns:

        metrics_per_epoch=metrics_df[~metrics_df['val_avg_loss'].isna()]
        metrics_per_epoch['epoch']=range(1,metrics_per_epoch.shape[0]+1)

    fig = go.Figure()

    if 'val_accuracy' in metrics_per_epoch.columns:

        fig.add_trace(go.Scatter(
            y=metrics_per_epoch['val_accuracy'],
            x=metrics_per_epoch['epoch'],
            mode='lines+markers+text',
            name='val_accuracy',
            text=metrics_per_epoch['val_accuracy'].round(2),  # Add labels to the points
            textposition='top center'  # Position the labels
        ))

    elif 'val_balanced_accuracy' in metrics_per_epoch.columns:
        
        fig.add_trace(go.Scatter(
            y=metrics_per_epoch['val_balanced_accuracy'],
            x=metrics_per_epoch['epoch'],
            mode='lines+markers+text',
            name='val_balanced_accuracy',
            text=metrics_per_epoch['val_balanced_accuracy'].round(2),  # Add labels to the points
            textposition='top center'  # Position the labels
        ))

    if 'total_train_accuracy' in metrics_per_epoch.columns:
        fig.add_trace(go.Scatter(
            y=metrics_per_epoch['total_train_accuracy'],
            x=metrics_per_epoch['epoch'],
            mode='lines+markers+text',
            name='train_accuracy',
            text=metrics_per_epoch['total_train_accuracy'].round(2),  # Add labels to the points
            textposition='top center'  # Position the labels
        ))

    elif 'train_balanced_accuracy' in metrics_per_epoch.columns:
            fig.add_trace(go.Scatter(
            y=metrics_per_epoch['train_balanced_accuracy'],
            x=metrics_per_epoch['epoch'],
            mode='lines+markers+text',
            name='train_balanced_accuracy',
            text=metrics_per_epoch['train_balanced_accuracy'].round(2),  # Add labels to the points
            textposition='top center'  # Position the labels
        ))


    fig.update_layout(
        title='Balanced Accuracy per Epoch', 
        #xaxis=dict(range=[0.9, 10.1],dtick=1),
        #yaxis=dict(range=[0.3, 1],dtick=0.05),
        xaxis_title='Epoch',
        yaxis_title='Balanced Accuracy'
    )

    # Center the title
    fig.update_layout(title={'x': 0.5})

    fig.show()


def loss_by_epoch(metrics_df):

    if 'val_loss' in metrics_df.columns:

        metrics_per_epoch=metrics_df[~metrics_df['val_loss'].isna()]

    elif 'val_avg_loss' in metrics_df.columns:

        metrics_per_epoch=metrics_df[~metrics_df['val_avg_loss'].isna()]
        metrics_per_epoch['epoch']=range(1,metrics_per_epoch.shape[0]+1)

    fig = go.Figure()

    if 'val_loss' in metrics_per_epoch.columns:

        fig.add_trace(go.Scatter(
            y=metrics_per_epoch['val_loss'],
            x=metrics_per_epoch['epoch'],
            mode='lines+markers+text',
            name='val_loss',
            text=metrics_per_epoch['val_loss'].round(2),  # Add labels to the points
            textposition='top center'  # Position the labels
        ))

    elif 'val_avg_loss' in metrics_per_epoch.columns:
        
        fig.add_trace(go.Scatter(
            y=metrics_per_epoch['val_avg_loss'],
            x=metrics_per_epoch['epoch'],
            mode='lines+markers+text',
            name='val_loss',
            text=metrics_per_epoch['val_avg_loss'].round(2),  # Add labels to the points
            textposition='top center'  # Position the labels
        ))

    if 'total_train_loss' in metrics_per_epoch.columns:
        fig.add_trace(go.Scatter(
            y=metrics_per_epoch['total_train_loss'],
            x=metrics_per_epoch['epoch'],
            mode='lines+markers+text',
            name='train_loss',
            text=metrics_per_epoch['total_train_loss'].round(2),  # Add labels to the points
            textposition='top center'  # Position the labels
        ))

    elif 'train_avg_loss' in metrics_per_epoch.columns:
            fig.add_trace(go.Scatter(
            y=metrics_per_epoch['train_avg_loss'],
            x=metrics_per_epoch['epoch'],
            mode='lines+markers+text',
            name='train_loss',
            text=metrics_per_epoch['train_avg_loss'].round(2),  # Add labels to the points
            textposition='top center'  # Position the labels
        ))


    fig.update_layout(
        title='Loss per Epoch', 
        #xaxis=dict(range=[0.9, 10.1],dtick=1),
        #yaxis=dict(range=[0.3, 1],dtick=0.05),
        xaxis_title='Epoch',
        yaxis_title='Loss'
    )

    # Center the title
    fig.update_layout(title={'x': 0.5})

    fig.show()


def balanced_accuracy_per_epoch_2_experiments(experiment_1,experiment_2):

    metrics_df_1=pd.read_csv(f'./experiments/{experiment_1}/metrics_df_{experiment_1}.csv')

    metrics_per_epoch_1=metrics_df_1[~metrics_df_1['val_avg_loss'].isna()]
    metrics_per_epoch_1['epoch']=range(1,metrics_per_epoch_1.shape[0]+1)

    fig = go.Figure()
        
    fig.add_trace(go.Scatter(
        y=metrics_per_epoch_1['val_balanced_accuracy'],
        x=metrics_per_epoch_1['epoch'],
        mode='lines+markers+text',
        name=f'val_balanced_accuracy_{experiment_1} (piano only)',
        text=metrics_per_epoch_1['val_balanced_accuracy'].round(2),  # Add labels to the points
        textposition='top center',  # Position the labels
        line=dict(color='lightblue')
    ))


    fig.add_trace(go.Scatter(
    y=metrics_per_epoch_1['train_balanced_accuracy'],
    x=metrics_per_epoch_1['epoch'],
    mode='lines+markers+text',
    name=f'train_balanced_accuracy_{experiment_1} (piano only)',
    text=metrics_per_epoch_1['train_balanced_accuracy'].round(2),  # Add labels to the points
    textposition='top center',  # Position the labels
    line=dict(color='#FF9999') 

    ))



    metrics_df_2=pd.read_csv(f'./experiments/{experiment_2}/metrics_df_{experiment_2}.csv')


    metrics_per_epoch_2=metrics_df_2[~metrics_df_2['val_avg_loss'].isna()]
    metrics_per_epoch_2['epoch']=range(1,metrics_per_epoch_2.shape[0]+1)


    fig.add_trace(go.Scatter(
        y=metrics_per_epoch_2['val_balanced_accuracy'],
        x=metrics_per_epoch_2['epoch'],
        mode='lines+markers+text',
        name=f'val_balanced_accuracy_{experiment_2} (full score)',
        text=metrics_per_epoch_2['val_balanced_accuracy'].round(2),  # Add labels to the points
        textposition='top center',  # Position the labels
        line=dict(color='darkblue')
    ))


    fig.add_trace(go.Scatter(
    y=metrics_per_epoch_2['train_balanced_accuracy'],
    x=metrics_per_epoch_2['epoch'],
    mode='lines+markers+text',
    name=f'train_balanced_accuracy_{experiment_2} (full score)',
    text=metrics_per_epoch_2['train_balanced_accuracy'].round(2),  # Add labels to the points
    textposition='top center',  # Position the labels
    line=dict(color='darkred')
    ))


    fig.update_layout(
        title='Balanced Accuracy per Epoch', 
        #xaxis=dict(range=[0.9, 10.1],dtick=1),
        #yaxis=dict(range=[0.3, 1],dtick=0.05),
        xaxis_title='Epoch',
        yaxis_title='Balanced Accuracy'
    )

    # Center the title
    fig.update_layout(title={'x': 0.5})

    fig.show()


def loss_per_epoch_2_experiments(experiment_1,experiment_2):

    metrics_df_1=pd.read_csv(f'./experiments/{experiment_1}/metrics_df_{experiment_1}.csv')

    metrics_per_epoch_1=metrics_df_1[~metrics_df_1['val_avg_loss'].isna()]
    metrics_per_epoch_1['epoch']=range(1,metrics_per_epoch_1.shape[0]+1)

    fig = go.Figure()
        
    fig.add_trace(go.Scatter(
        y=metrics_per_epoch_1['val_avg_loss'],
        x=metrics_per_epoch_1['epoch'],
        mode='lines+markers+text',
        name=f'val_loss_{experiment_1} (piano only)',
        text=metrics_per_epoch_1['val_avg_loss'].round(2),  # Add labels to the points
        textposition='top center',  # Position the labels
        line=dict(color='lightblue')
    ))


    fig.add_trace(go.Scatter(
    y=metrics_per_epoch_1['train_avg_loss'],
    x=metrics_per_epoch_1['epoch'],
    mode='lines+markers+text',
    name=f'train_loss_{experiment_1} (piano only)',
    text=metrics_per_epoch_1['train_avg_loss'].round(2),  # Add labels to the points
    textposition='top center',  # Position the labels
    line=dict(color='#FF9999') 

    ))



    metrics_df_2=pd.read_csv(f'./experiments/{experiment_2}/metrics_df_{experiment_2}.csv')


    metrics_per_epoch_2=metrics_df_2[~metrics_df_2['val_avg_loss'].isna()]
    metrics_per_epoch_2['epoch']=range(1,metrics_per_epoch_2.shape[0]+1)


    fig.add_trace(go.Scatter(
        y=metrics_per_epoch_2['val_avg_loss'],
        x=metrics_per_epoch_2['epoch'],
        mode='lines+markers+text',
        name=f'val_loss_{experiment_2} (full score)',
        text=metrics_per_epoch_2['val_avg_loss'].round(2),  # Add labels to the points
        textposition='top center',  # Position the labels
        line=dict(color='darkblue')
    ))


    fig.add_trace(go.Scatter(
    y=metrics_per_epoch_2['train_avg_loss'],
    x=metrics_per_epoch_2['epoch'],
    mode='lines+markers+text',
    name=f'train_loss_{experiment_2} (full score)',
    text=metrics_per_epoch_2['train_avg_loss'].round(2),  # Add labels to the points
    textposition='top center',  # Position the labels
    line=dict(color='darkred')
    ))


    fig.update_layout(
        title='Loss per Epoch', 
        #xaxis=dict(range=[0.9, 10.1],dtick=1),
        #yaxis=dict(range=[0.3, 1],dtick=0.05),
        xaxis_title='Epoch',
        yaxis_title='Loss'
    )

    # Center the title
    fig.update_layout(title={'x': 0.5})

    fig.show()


def to_bold(text):
    bold_map = {
        'A': '𝗔', 'B': '𝗕', 'C': '𝗖', 'D': '𝗗', 'E': '𝗘', 'F': '𝗙', 'G': '𝗚',
        'H': '𝗛', 'I': '𝗜', 'J': '𝗝', 'K': '𝗞', 'L': '𝗟', 'M': '𝗠', 'N': '𝗡',
        'O': '𝗢', 'P': '𝗣', 'Q': '𝗤', 'R': '𝗥', 'S': '𝗦', 'T': '𝗧', 'U': '𝗨',
        'V': '𝗩', 'W': '𝗪', 'X': '𝗫', 'Y': '𝗬', 'Z': '𝗭',
        'a': '𝗮', 'b': '𝗯', 'c': '𝗰', 'd': '𝗱', 'e': '𝗲', 'f': '𝗳', 'g': '𝗴',
        'h': '𝗵', 'i': '𝗶', 'j': '𝗷', 'k': '𝗸', 'l': '𝗹', 'm': '𝗺', 'n': '𝗻',
        'o': '𝗼', 'p': '𝗽', 'q': '𝗾', 'r': '𝗿', 's': '𝘀', 't': '𝘁', 'u': '𝘂',
        'v': '𝘃', 'w': '𝘄', 'x': '𝘅', 'y': '𝘆', 'z': '𝘇'
    }
    return ''.join(bold_map.get(char, char) for char in text)


def results_table(export_data=False):

    experiments_description=['piano scores, 3 epochs, REMI tokenizer', #1
                            'piano scores, 3 epochs, REMI tokenizer', #2
                            'piano scores, 10 epochs, REMI tokenizer', #3
                            'piano scores, 20 epochs, REMI tokenizer', #4
                            'type0 scores, 10 epochs, REMI tokenizer', #5
                            'piano scores, 10 epochs, TSD tokenizer', #6
                            'piano scores, 10 epochs, REMI tokenizer,shuffle', #7
                            'piano scores, 50 epochs, REMI tokenizer', #8
                            'piano scores, 10 epochs,REMI+MLP', #9
                            'type0 scores, 10 epochs,REMI+MLP', #10
                            'merged piano scores,10 epochs,REMI+MLP', #11
                            'voice only scores, 10 epochs,REMI+MLP', #12
                            'left hand scores, 10 epochs,REMI+MLP', #13
                            'right hand scores, 10 epochs,REMI+MLP' #14
                            ]
    
    ##============================== VALIDATION DATA =================================##
    full_metrics_df_val=pd.DataFrame([])

    for number in range(1,15):
        df_val_temp=pd.read_csv(f'./experiments/e{number}/predictions_df_val_e{number}.csv')
        df_val_temp['experiment_number']=f'e{number}'
        full_metrics_df_val=pd.concat([full_metrics_df_val,df_val_temp])

    full_metrics_df_val=full_metrics_df_val.reset_index(drop=True).drop(columns='Unnamed: 0')

    rows=[]

    for number in range(1,15):
        y_true=full_metrics_df_val[full_metrics_df_val['experiment_number']==f'e{number}']['labels']
        y_pred=full_metrics_df_val[full_metrics_df_val['experiment_number']==f'e{number}']['predictions']
        
        # Create a new row for the current experiment
        new_row = {'experiment': f'e{number}',
                'val_balanced_accuracy_score': round(balanced_accuracy_score(y_true=y_true,y_pred=y_pred),2),
                'val_accuracy': round(accuracy_score(y_true=y_true,y_pred=y_pred),2),
                'val_f1_score':round(f1_score(y_true=y_true,y_pred=y_pred),2)}
        
        # Append the new row to the DataFramebalanced_accuracy_df_test
        rows.append(new_row)

    metrics_df_val=pd.DataFrame(rows)
    metrics_df_val['description']=experiments_description

    
    ##===================================TEST DATA==================================##
    full_metrics_df_test=pd.DataFrame([])

    for number in range(1,15):
        df_test_temp=pd.read_csv(f'./experiments/e{number}/predictions_df_test_e{number}.csv')
        df_test_temp['experiment_number']=f'e{number}'
        full_metrics_df_test=pd.concat([full_metrics_df_test,df_test_temp])

    full_metrics_df_test=full_metrics_df_test.reset_index(drop=True).drop(columns='Unnamed: 0')

    rows=[]

    for number in range(1,15):
        y_true=full_metrics_df_test[full_metrics_df_test['experiment_number']==f'e{number}']['labels']
        y_pred=full_metrics_df_test[full_metrics_df_test['experiment_number']==f'e{number}']['predictions']
        
        # Create a new row for the current experiment
        new_row = {'experiment': f'e{number}',
                'test_balanced_accuracy_score': round(balanced_accuracy_score(y_true=y_true,y_pred=y_pred),2),
                'test_accuracy': round(accuracy_score(y_true=y_true,y_pred=y_pred),2),
                'test_f1_score':round(f1_score(y_true=y_true,y_pred=y_pred),2)}
        
        # Append the new row to the DataFrame
        rows.append(new_row)

    metrics_df_test=pd.DataFrame(rows)

    ##================================= TRAIN DATA ===============================================##
    full_metrics_df_train=pd.DataFrame([])

    for number in range(1,11):
        df_train_temp=pd.read_csv(f'./experiments/e{number}/predictions_df_train_e{number}.csv')
        df_train_temp['experiment_number']=f'e{number}'
        full_metrics_df_train=pd.concat([full_metrics_df_train,df_train_temp])

    full_metrics_df_train=full_metrics_df_train.reset_index(drop=True).drop(columns='Unnamed: 0')

    rows=[]

    for number in range(1,11):
        y_true=full_metrics_df_train[full_metrics_df_train['experiment_number']==f'e{number}']['labels']
        y_pred=full_metrics_df_train[full_metrics_df_train['experiment_number']==f'e{number}']['predictions']
        
        # Create a new row for the current experiment
        new_row = {'experiment': f'e{number}',
                'train_balanced_accuracy_score': round(balanced_accuracy_score(y_true=y_true,y_pred=y_pred),2),
                'train_accuracy': round(accuracy_score(y_true=y_true,y_pred=y_pred),2),
                'train_f1_score': round(f1_score(y_true=y_true,y_pred=y_pred),2)}
        
        # Append the new row to the DataFrame
        rows.append(new_row)
    
    for number in range(12,15):
        df_train_temp=pd.read_csv(f'./experiments/e{number}/metrics_df_e{number}.csv')
        train_balanced_accuracy=df_train_temp.loc[df_train_temp.index[-1],'train_balanced_accuracy']
        
        # Create a new row for the current experiment
        new_row = {'experiment': f'e{number}',
                'train_balanced_accuracy_score': train_balanced_accuracy,
                'train_accuracy': '',
                'train_f1_score': ''}
        
        # Append the new row to the DataFrame
        rows.append(new_row)

    metrics_df_train=pd.DataFrame(rows)

    ##============ CREATING FULL METRICS DATASET =========================##

    full_metrics_df=metrics_df_val.merge(metrics_df_test, on='experiment', how='left').merge(metrics_df_train, on='experiment', how='left')
    full_metrics_df.fillna('',inplace=True)

    full_metrics_df=full_metrics_df[['experiment', 'description',
                                    'train_balanced_accuracy_score', 'train_accuracy','train_f1_score',
                                    'val_balanced_accuracy_score', 'val_accuracy', 'val_f1_score',
                                    'test_balanced_accuracy_score', 'test_accuracy','test_f1_score', ]]

    if export_data==True:
        full_metrics_df.to_csv('experiments_metrics.csv',index=False)

    return full_metrics_df


def prepare_validation_data(predictions_df_val_path,validation_set_path):

    df_temp=pd.read_csv(predictions_df_val_path)

    if 'epoch' in df_temp.columns:
        df_temp=df_temp[df_temp['epoch']==max(df_temp['epoch'])]

    val_set=pd.read_csv(validation_set_path)

    val_set['binary_label']= val_set['composer_gender'].apply(lambda x: 0 if x == 'Male' else 1)
    df_temp['predictions_string']= df_temp['predictions'].apply(lambda x: 'Male' if x == 0 else 'Female')

    # full_df=pd.DataFrame(columns=['path', 'name', 'set_id', 'composer_path', 'composer_name',
    #    'composer_gender', 'desc', 'sets', 'scores', 'scores_paths',
    #    'piano_scores_paths', 'contains_piano_track?', 'binary_label',
    #    'Unnamed: 0', 'predictions', 'labels', 'epoch','predictions_string'])

    # for epoch in set(df_temp['epoch']):

    #     epoch_df=df_temp[df_temp['epoch']==epoch]
    #     concat_temp=pd.concat([epoch_df.reset_index(drop=True),val_set.reset_index(drop=True)],axis=1)
    
    full_df=pd.concat([df_temp,val_set],axis=1)

    full_df=full_df.groupby(by=['composer_name','predictions_string'])['predictions'].count().reset_index()

    # Calculate total predictions for each composer
    full_df['total_predictions'] = full_df.groupby('composer_name')['predictions'].transform('sum')

    # Calculate the proportion of predictions
    full_df['proportion'] = round(full_df['predictions'] / full_df['total_predictions'],2)

    composer_gender=val_set[['composer_name', 'composer_gender']].drop_duplicates()

    full_df=full_df.merge(composer_gender, on='composer_name',how='left')
    
    return full_df


def prepare_test_data(predictions_df_test_path,test_set_path):

    df_temp=pd.read_csv(predictions_df_test_path)

    if 'epoch' in df_temp.columns:
        df_temp=df_temp[df_temp['epoch']==max(df_temp['epoch'])]

    test_set=pd.read_csv(test_set_path)

    test_set['binary_label']= test_set['composer_gender'].apply(lambda x: 0 if x == 'Male' else 1)
    df_temp['predictions_string']= df_temp['predictions'].apply(lambda x: 'Male' if x == 0 else 'Female')

    full_df=pd.concat([test_set,df_temp],axis=1)

    full_df=full_df.groupby(by=['composer_name','predictions_string'])['predictions'].count().reset_index()

    # Calculate total predictions for each composer
    full_df['total_predictions'] = full_df.groupby('composer_name')['predictions'].transform('sum')

    # Calculate the proportion of predictions
    full_df['proportion'] = round(full_df['predictions'] / full_df['total_predictions'],2)

    composer_gender=test_set[['composer_name', 'composer_gender']].drop_duplicates()

    full_df=full_df.merge(composer_gender, on='composer_name',how='left')

    return full_df


def conditional_probabilities_by_gender(full_df):


    grouped_df=full_df[['composer_gender','predictions_string','predictions']].groupby(by=['composer_gender','predictions_string']).sum().reset_index()

    # Calculate the total predictions for each gender
    total_predictions = grouped_df.groupby('composer_gender')['predictions'].transform('sum')

    # Calculate the proportion of each prediction string within each gender
    grouped_df['proportion'] = round(grouped_df['predictions'] / total_predictions,2)

    # # Create the bar chart
    # fig = px.bar(grouped_df,
    #             x='composer_gender',
    #             y='proportion',
    #             color='predictions_string',
    #             barmode='group',                
    #             labels={'composer_gender': to_bold('True')+' Composer Gender',
    #                      'proportion': 'Predictions Proportion',
    #                      'predictions_string': to_bold('Predicted')+' composer gender'},                        
    #             color_discrete_map={'Female': 'red', 'Male': 'blue'},
    #             title='Proportion of Predictions by gender',
    #             text_auto=True
                
    #             )

    # # Center the title
    # fig.update_layout(title={'x': 0.5})

    # # Show the plot
    # fig.show()

    # Create the bar chart
    fig = px.bar(grouped_df,
                x='composer_gender',
                y='proportion',
                color='predictions_string',
                barmode='group',
                labels={
                    'composer_gender': '<b>True Composer Gender</b>',
                    'proportion': 'Predictions Proportion',
                    'predictions_string': 'Predicted Composer Gender'
                },
                color_discrete_map={'Female': 'red', 'Male': 'blue'},
                title='<b>Proportion of Predictions by Gender</b>',
                text_auto=True
                )

    # Center the title and add padding
    fig.update_layout(
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20),
            'pad': dict(t=30)
        },
        xaxis=dict(
            title='<b>True Composer Gender</b>',  # Explicitly set the x-axis title
            tickfont=dict(size=12),
            titlefont=dict(size=16)
        ),
        yaxis=dict(
            title='Predictions Proportion',
            gridcolor='lightgrey',
            tickfont=dict(size=12),
            titlefont=dict(size=16),
            automargin=True
        ),

        plot_bgcolor='white',
        width=800,
        height=600,

        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.3,  # Adjust the legend's y position to provide more space for the x-axis title
            xanchor='center',
            yanchor='top',
            font=dict(size=10)
        ),

        margin=dict(l=80, r=50, t=100, b=120)  # Increase the bottom margin to provide space for the x-axis title
        ,
        font=dict(
            family="Times New Roman, Times, serif",
            size=12,
            color="black"
        )
    )

    # Show the plot
    fig.show()

def proportions_male_composers(full_df):
        # Create the bar chart
        fig = px.bar(full_df[full_df['composer_gender']=='Male'],
                x='composer_name',
                y='proportion',
                color='predictions_string',
                barmode='group',
                labels={'composer_name':to_bold('Male')+' Composer', 
                        'proportion': 'Proportion of predictions',
                        'predictions_string': 'Predicted Gender'},
                color_discrete_map={'Female': 'red', 'Male': 'blue'},
                title= f'Proportion of Predictions for {to_bold('MALE')} composers',
                text_auto=True)


        # Center the title
        fig.update_layout(title={'x': 0.5})

        # Show the plot
        fig.show()


def proportions_female_composers(full_df):
        # Create the bar chart
        fig = px.bar(full_df[full_df['composer_gender']=='Female'],
                x='composer_name',
                y='proportion',
                color='predictions_string',
                barmode='group',
                labels={'composer_name': to_bold('Female')+' Composer', 
                        'proportion': 'Proportion of predictions',
                        'predictions_string': 'Predicted Gender'},
                color_discrete_map={'Female': 'red', 'Male': 'blue'},
                title= f'Proportion of Predictions for {to_bold('FEMALE')} composers',
                text_auto=True)


        # Center the title
        fig.update_layout(title={'x': 0.5})

        # Show the plot
        fig.show()


def proportions_female_composers_and_nscores(full_df):

    # Calculate the total predictions for each composer
    total_predictions = full_df[full_df['composer_gender']=='Female'].groupby('composer_name')['predictions'].sum().reset_index()
    total_predictions.rename(columns={'predictions': 'total_predictions_sum'}, inplace=True)

    # Sort both dataframes by composer_name
    sorted_df = full_df[full_df['composer_gender']=='Female'].sort_values(by='composer_name')
    total_predictions = total_predictions.sort_values(by='composer_name')

    # Create the bar chart
    fig = px.bar(sorted_df, x='composer_name', y='proportion', color='predictions_string', barmode='group',
                labels={'composer_name': to_bold('Female')+' Composers',
                        'proportion': 'Proportion of predictions',
                        'predictions_string': 'Predicted Gender'},
                title=f'Proportion of Gender Predictions by Composer ({to_bold("FEMALE")})',
                text='predictions_string',
                color_discrete_map={'Female': 'red', 'Male': 'blue'},
                text_auto=True)

    # Set the category order for the x-axis to ensure alphabetical sorting
    fig.update_xaxes(categoryorder='category ascending')

    # Create the secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title='Total Predictions',
            overlaying='y',
            side='right'
        ),
        yaxis=dict(
            title='Proportion of predictions'
        ),
        legend=dict(
            x=0.02,  # Adjust the x position of the legend
            y=1.5,   # Adjust the y position of the legend
            traceorder='normal',
            font=dict(
                family='sans-serif',
                size=12,
                color='black'
            ),
            bgcolor='rgba(0,0,0,0)',  # Transparent background
            bordercolor='rgba(0,0,0,0)'  # Transparent border
        ),
        margin=dict(l=100)  # Increase left margin to create space between y-axis and legend
    )

    # Add a line trace for total predictions
    fig.add_trace(
        go.Scatter(
            x=total_predictions['composer_name'],
            y=total_predictions['total_predictions_sum'],
            mode='markers+lines+text',
            name='Number of scores',
            yaxis='y2',  # Set this trace to use the secondary y-axis
            line=dict(color='orange'),
            marker=dict(size=10),
            text=[f"<b><span style='color:orange'>{value}</span></b>" for value in total_predictions['total_predictions_sum']],
            textposition='top center'
        )
    )

    # Center the title
    fig.update_layout(title={'x': 0.5})

    # Show the plot
    fig.show()


def proportions_male_composers_and_nscores(full_df):

    # Calculate the total predictions for each composer
    total_predictions = full_df[full_df['composer_gender']=='Male'].groupby('composer_name')['predictions'].sum().reset_index()
    total_predictions.rename(columns={'predictions': 'total_predictions_sum'}, inplace=True)

    # Sort both dataframes by composer_name
    sorted_df = full_df[full_df['composer_gender']=='Male'].sort_values(by='composer_name')
    total_predictions = total_predictions.sort_values(by='composer_name')

    # Create the bar chart
    fig = px.bar(sorted_df, x='composer_name', y='proportion', color='predictions_string', barmode='group',
                labels={'composer_name': to_bold('Male')+' Composers',
                        'proportion': 'Proportion of predictions',
                        'predictions_string': 'Predicted Gender'},
                title=f'Proportion of Gender Predictions by Composer ({to_bold("MALE")})',
                text='predictions_string',
                color_discrete_map={'Female': 'red', 'Male': 'blue'},
                text_auto=True)

    # Set the category order for the x-axis to ensure alphabetical sorting
    fig.update_xaxes(categoryorder='category ascending')

    # Create the secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title='Total Predictions',
            overlaying='y',
            side='right'
        ),
        yaxis=dict(
            title='Proportion of predictions'
        ),
        legend=dict(
            x=0.02,  # Adjust the x position of the legend
            y=1.5,   # Adjust the y position of the legend
            traceorder='normal',
            font=dict(
                family='sans-serif',
                size=12,
                color='black'
            ),
            bgcolor='rgba(0,0,0,0)',  # Transparent background
            bordercolor='rgba(0,0,0,0)'  # Transparent border
        ),
        margin=dict(l=100)  # Increase left margin to create space between y-axis and legend
    )

    # Add a line trace for total predictions
    fig.add_trace(
        go.Scatter(
            x=total_predictions['composer_name'],
            y=total_predictions['total_predictions_sum'],
            mode='markers+lines+text',
            name='Number of scores',
            yaxis='y2',  # Set this trace to use the secondary y-axis
            line=dict(color='orange'),
            marker=dict(size=10),
            text=[f"<b><span style='color:orange'>{value}</span></b>" for value in total_predictions['total_predictions_sum']],
            textposition='top center'
        )
    )

    # Center the title
    fig.update_layout(title={'x': 0.5})

    # Show the plot
    fig.show()


def density_female_composers(full_df):
    # Create the histogram with adjusted histnorm and bins
    fig = px.histogram(full_df[full_df['composer_gender'] == 'Female'], 
                    x='proportion', 
                    histnorm='probability',  # Adjusted parameter
                    color='predictions_string',
                    labels={'composer_name': to_bold('Female') + ' Composer', 
                            'proportion': 'Proportion of predictions',
                            'predictions_string': 'Predicted Gender'},              
                    marginal='box',
                    barmode='overlay',
                    color_discrete_map={'Female': 'red', 'Male': 'blue'},
                    title=f'Proportion of Predictions for {to_bold("FEMALE")} composers',
                    nbins=10  # Adjust number of bins if necessary
    )

    # Center the title
    fig.update_layout(title={'x': 0.5})

    # Show the plot
    fig.show()


def density_male_composers(full_df):
    # Create the histogram with adjusted histnorm and bins
    fig = px.histogram(full_df[full_df['composer_gender'] == 'Male'], 
                    x='proportion', 
                    histnorm='probability',  # Adjusted parameter
                    color='predictions_string',
                    labels={'composer_name': to_bold('Male') + ' Composer', 
                            'proportion': 'Proportion of predictions',
                            'predictions_string': 'Predicted Gender'},              
                    marginal='box',
                    barmode='overlay',
                    color_discrete_map={'Female': 'red', 'Male': 'blue'},
                    title=f'Proportion of Predictions for {to_bold("MALE")} composers',
                    nbins=10  # Adjust number of bins if necessary
    )

    # Center the title
    fig.update_layout(title={'x': 0.5})

    # Show the plot
    fig.show()


def e10_e14_validation_performance():
       
    rows=[]

    # Experiment names
    experiment_name = ["Full scores", "Only piano", "Only Voice", "Left hand Only", "Right hand Only"]


    for i,number in enumerate(range(10,15)):

        metrics_df_temp=pd.read_csv(f'./experiments/e{number}/metrics_df_e{number}.csv')
        new_row = {'experiment': experiment_name[i],
                'val_balanced_accuracy_score': round(metrics_df_temp.loc[metrics_df_temp.index[-1],'val_balanced_accuracy'],2)}
        rows.append(new_row)

    metrics_df_val=pd.DataFrame(rows)

    # Custom colors
    colors = ['darkred', 'lightblue', 'lightgreen', 'lightpink', 'lightyellow']

    # Additional text for labels
    #additional_text = ["Full scores", "Only piano", "Only Voice", "Left hand Only", "Right hand Only"]

    # Create the bar plot
    fig = px.bar(metrics_df_val, 
                x='experiment', 
                y='val_balanced_accuracy_score')

    # # Update the bars with custom colors and text
    # fig.update_traces(marker_color=colors, 
    #                #text=[f"{v} ({t})" for v, t in zip(metrics_df_val['val_balanced_accuracy_score'])]
    #                text=metrics_df_val['val_balanced_accuracy_score'],
    #                texttemplate='%{text}')
    
    # Update the bars with custom colors and text
    fig.update_traces(marker_color=colors,
                   text=metrics_df_val['val_balanced_accuracy_score'],
                   texttemplate='%{text}')

    # Add a title
    fig.update_layout(title_text="<b>Validation Balanced Accuracy by Score's feature",
                    yaxis_title='Validation Balanced Accuracy')

    fig.update_layout(title={'x':0.5},
                    
                        yaxis=dict(
                            gridcolor='lightgrey',
                            automargin=True
                        ),
                        plot_bgcolor='white',
                        width=800,
                        height=600,
                        font=dict(
                            family="DejaVu Sans",
                            size=16,
                            color="black"
                        ),
                        
                        margin=dict(l=0, r=0, t=80, b=0)
                        
        )   
        
    # Show the plot
    fig.show()

    return fig


def proportions_by_composer(df):      

    # Create columns for positive and negative values
    df['positive'] = df['proportion']
    df['negative'] = df.apply(lambda row: -(1 - row['proportion']), axis=1)
    df = df.sort_values(by='proportion')

    # Create enumeration and labels for y-axis
    df['enumeration'] = range(1, df.shape[0] + 1)
    enumerated_labels = [f" {name} - {i} " for i, name in zip(df['enumeration'], df['composer_name'])]

    # Create a color mapping based on the composer_gender
    color_map = {'Female': 'red', 'Male': 'blue'}
    df['color'] = df['composer_gender'].map(color_map)

    # Create traces for positive and negative values
    trace_positive = go.Bar(
        y=df['composer_name'],
        x=df['positive'],
        text=df['enumeration'],
        textposition='inside',
        marker_color=df['color'],
        orientation='h',
        showlegend=False,
        name='Positive Proportion'
    )

    trace_negative = go.Bar(
        y=df['composer_name'],
        x=df['negative'],
        textposition='inside',
        marker_color=df['color'],
        orientation='h',
        showlegend=False,
        name='Negative Proportion'
    )

    df.reset_index(inplace=True, drop=True)
    # Identify the indices for 50-50% composers
    lower_bound_index = df[df['positive'] == 0.500000].index[0]
    upper_bound_index = df[df['positive'] == 0.500000].index[-1]

    # Create the figure
    fig = go.Figure(data=[trace_positive, trace_negative])

    # Add reference lines at 50% and -50%
    fig.add_shape(type="line", 
                x0=0.5, y0=-0.5, x1=0.5, y1=len(df) - 0.5,
                line=dict(color="black", width=1, dash="dash"))
    fig.add_shape(type="line", 
                x0=-0.5, y0=-0.5, x1=-0.5, y1=len(df) - 0.5,
                line=dict(color="black", width=1, dash="dash"))

    # Add horizontal lines for for 50-50% composers
    fig.add_shape(type="line",
                x0=-1, y0=lower_bound_index - 0.5, x1=1, y1=lower_bound_index - 0.5,
                line=dict(color="green", width=2, dash="dash"))

    fig.add_shape(type="line",
                x0=-1, y0=upper_bound_index + 0.5, x1=1, y1=upper_bound_index + 0.5,
                line=dict(color="green", width=2, dash="dash"))

    # Create custom x-tick labels
    custom_x_ticks = {v: f"{abs(v) * 100:.0f}%" for v in [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]}

    # Update the layout
    fig.update_layout(
        barmode='overlay',
        width=800,  # Increase the width of the plot
        height=1300,
        title=dict(
            text='<b>Majority vote Predictions by Composer',
            font=dict(size=16, family='DejaVu Sans',color='black'),
            x=0.5,  # Center the title
            xanchor='center'
        ),
        yaxis_title=None,
        xaxis_title=None,  # Remove x-axis title
        yaxis=dict(
            title=None,
            tickfont_size=12,
            tickmode='array',
            tickvals=df['composer_name'],  # Y-axis positions
            ticktext=enumerated_labels,  # Y-axis labels with enumeration
            tickfont=dict(family='DejaVu Sans',color='black')
        ),
        xaxis=dict(
            title_font_size=20,
            tickfont_size=12,
            tickvals=list(custom_x_ticks.keys()),
            ticktext=list(custom_x_ticks.values()),
            tickformat='.0%',
            range=[-1, 1],
            title=None,  # Remove x-axis title
            tickfont=dict(family='DejaVu Sans',color='black')
        ),
        margin=dict(l=200, r=20, t=100, b=50)  # Adjust margins to fit labels and avoid extra space
    )

    # Add annotations for positive and negative x-axis titles
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.1, y=-0.035,  # Adjust x for positioning near the x-axis
        text="% of Male predictions",
        showarrow=False,
        font=dict(size=12, family='DejaVu Sans', color='black'),
        align="center"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.9, y=-0.035,  # Adjust x for positioning near the x-axis
        text="% of Female predictions",
        showarrow=False,
        font=dict(size=12, family='DejaVu Sans', color='black'),
        align="center"
    )

    import numpy as np
    # Define conditions
    conditions = [
        (df['positive'] == -df['negative']),   # Condition 1: 50-50%
        (df['positive'] > -df['negative']),    # Condition 2: Positive is greater
        (df['positive'] < -df['negative'])     # Condition 3: Negative is greater
    ]

    # Define choices for each condition
    choices = ['50-50%', 'Predicted Female', 'Predicted Male']

    # Create the new column based on conditions
    df['category'] = np.select(conditions, choices, default='Other')

    count_by_category=df.groupby(by=['category','composer_gender'])['predictions'].count().reset_index()

    # Define the positions and text for the annotations with titles on top
    annotations = [
        dict(
            xref="paper", yref="paper",
            x=0, y=0.999,
            text=f"{to_bold('Predicted Female')}<br><span style='color:blue'>■</span> Male: {count_by_category[(count_by_category['category']=='Predicted Female')&(count_by_category['composer_gender']=='Male')]['predictions'].values[0]}<br><span style='color:red'>■</span> Female: {count_by_category[(count_by_category['category']=='Predicted Female')&(count_by_category['composer_gender']=='Female')]['predictions'].values[0]}",
            showarrow=False,
            font=dict(size=12, family='DejaVu Sans', color='black'),
            align="left"
        ),
        dict(
            xref="paper", yref="paper",
            x=0, y=0.501,
            text=f"{to_bold('Tie')}<br><span style='color:blue'>■</span> Male: {count_by_category[(count_by_category['category']=='50-50%')&(count_by_category['composer_gender']=='Male')]['predictions'].values[0]}<br><span style='color:red'>■</span> Female: {count_by_category[(count_by_category['category']=='50-50%')&(count_by_category['composer_gender']=='Female')]['predictions'].values[0]}",
            showarrow=False,
            font=dict(size=12, family='DejaVu Sans', color='black'),
            align="left"
        ),
        dict(
            xref="paper", yref="paper",
            x=0, y=0.38,
            text=f"{to_bold('Predicted Male')}<br><span style='color:blue'>■</span> Male: {count_by_category[(count_by_category['category']=='Predicted Male')&(count_by_category['composer_gender']=='Male')]['predictions'].values[0]}<br><span style='color:red'>■</span> Female: {count_by_category[(count_by_category['category']=='Predicted Male')&(count_by_category['composer_gender']=='Female')]['predictions'].values[0]}",
            showarrow=False,
            font=dict(size=12, family='DejaVu Sans', color='black'),
            align="left"
        )
    ]

    # Add the annotations to the figure
    for annotation in annotations:
        fig.add_annotation(annotation)

    # Add a legend-like annotation below the plot
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.7, y=1.025,  # Position the legend below the plot
        text="True Composer Gender  </b> <span style='color:blue'>■</span> Male <span style='color:red'>■</span> Female",
        showarrow=False,
        font=dict(size=12, family='DejaVu Sans', color='black'),
        align="center",
        xanchor='center'
    )

    # Show the figure
    fig.show()

    return fig



    ##=============vertical, grouped plot============##

# df['positive'] = df['proportion']
# df['negative'] = df.apply(lambda row: -(1 - row['proportion']), axis=1)
# df=df.sort_values(by='proportion')

# # Create enumeration and labels for y-axis
# df['enumeration'] = df.groupby('composer_gender').cumcount() + 1
# enumerated_labels = [f" {name} - {i} " for i, name in zip(df['enumeration'], df['composer_name'])]

# # Create a color mapping based on the composer_gender
# color_map = {'Female': 'red', 'Male': 'blue'}

# # Create traces for positive and negative values
# traces = []

# for gender, color in color_map.items():

#     df_gender = df[df['composer_gender'] == gender]

#     trace_positive = go.Bar(
#         y=df_gender['composer_name'],
#         x=df_gender['positive'],
#         name=gender,
#         text=df_gender['enumeration'],
#         textposition='inside',
#         marker_color=color,
#         orientation='h'
#     )

#     trace_negative = go.Bar(
#         y=df_gender['composer_name'],
#         x=df_gender['negative'],
#         textposition='inside',
#         marker_color=color,
#         orientation='h',
#         showlegend=False
#     )

#     traces.extend([trace_positive, trace_negative])

# # Create the figure
# fig = go.Figure(data=traces)

# # Add reference lines at 50% and -50%
# fig.add_shape(type="line", 
#               x0=0.5, y0=-0.5, x1=0.5, y1=len(df)-0.5,
#               line=dict(color="black", width=1, dash="dash"))
# fig.add_shape(type="line", 
#               x0=-0.5, y0=-0.5, x1=-0.5, y1=len(df)-0.5,
#               line=dict(color="black", width=1, dash="dash"))

# # Create custom x-tick labels
# custom_x_ticks = {v: f"{abs(v)*100:.0f}%" for v in [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]}

# # Update the layout
# fig.update_layout(
#     barmode='overlay',
#     width=800,  # Increase the width of the plot
#     height=1300,
#     title=dict(
#         text='Proportion of Female and Male Predictions by Composer',
#         font=dict(size=16, family='Times New Roman'),
#         x=0.5,  # Center the title
#         xanchor='center'
#     ),
#     yaxis_title=None,
#     xaxis_title=None,  # Remove x-axis title
#     yaxis=dict(
#         title=None, 
#         tickfont_size=14,
#         tickmode='array',
#         tickvals=df['composer_name'],  # Y-axis positions
#         ticktext=enumerated_labels,  # Y-axis labels with enumeration
#         tickfont=dict(family='Times New Roman')
#     ),
#     xaxis=dict(
#         title_font_size=20,
#         tickfont_size=14,
#         tickvals=list(custom_x_ticks.keys()),
#         ticktext=list(custom_x_ticks.values()),
#         tickformat='.0%',
#         range=[-1, 1],
#         title=None,  # Remove x-axis title
#         tickfont=dict(family='Times New Roman')
#     ),
#     legend=dict(
#         title='True Composer Gender',
#         orientation="h",
#         yanchor="top",
#         y=1.04,
#         xanchor="center",
#         x=0.7,
#         font=dict(size=12, family='Times New Roman')
#     ),
#     margin=dict(l=200, r=20, t=100, b=50)  # Adjust margins to fit labels and avoid extra space
# )

# # Add annotations for positive and negative x-axis titles
# fig.add_annotation(
#     xref="paper", yref="paper",
#     x=0.1, y=-0.035,  # Adjust x for positioning near the x-axis
#     text="% of Male predictions",
#     showarrow=False,
#     font=dict(size=16, family='Times New Roman'),
#     align="center"
# )

# fig.add_annotation(
#     xref="paper", yref="paper",
#     x=0.9, y=-0.035,  # Adjust x for positioning near the x-axis
#     text="% of Female predictions",
#     showarrow=False,
#     font=dict(size=16, family='Times New Roman'),
#     align="center"
# )

# # Show the figure
# fig.show()

