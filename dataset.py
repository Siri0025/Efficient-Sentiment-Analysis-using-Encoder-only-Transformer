import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import os


save_dir = '/home/careinfolab/Rohan/rohanj/Transformers/Group7Proj'
csv_file_twitter = r'/home/careinfolab/Rohan/rohanj/Transformers/Group7Proj/processed_twitter.csv'
csv_file_linkedin = f'/home/careinfolab/Rohan/rohanj/Transformers/Group7Proj/processed_linkedin.csv'

# Load the data
twitter_df = pd.read_csv(csv_file_twitter)
linkedin_df = pd.read_csv(csv_file_linkedin)

# Tokenize the cleaned text
twitter_df_token_texts = twitter_df['cleaned_text'].apply(lambda x: x.split())
linkedin_df_token_texts = linkedin_df['cleaned_text'].apply(lambda x: x.split())


w2v_model_twitter = Word2Vec(
    sentences=twitter_df_token_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)
w2v_model_linkedin = Word2Vec(
    sentences=linkedin_df_token_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

def get_sentence_vector(tokens, model):
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

twitter_embeddings = twitter_df_token_texts.apply(lambda tokens: get_sentence_vector(tokens, w2v_model_twitter))
linkedin_embeddings = linkedin_df_token_texts.apply(lambda tokens: get_sentence_vector(tokens, w2v_model_linkedin))

twitter_embeddings_df = pd.DataFrame(twitter_embeddings.tolist(), columns=[f'embedding_{i}' for i in range(w2v_model_twitter.vector_size)])
linkedin_embeddings_df = pd.DataFrame(linkedin_embeddings.tolist(), columns=[f'embedding_{i}' for i in range(w2v_model_linkedin.vector_size)])

twitter_final_df = pd.concat([twitter_embeddings_df, twitter_df['sentiment']], axis=1)
twitter_final_df['seq_length'] = twitter_df_token_texts.apply(len)

linkedin_final_df = pd.concat([linkedin_embeddings_df, linkedin_df['sentiment']], axis=1)
linkedin_final_df['seq_length'] = linkedin_df_token_texts.apply(len)

if 'seq_length' in twitter_final_df.columns:
    print(twitter_final_df['seq_length'].max(skipna=True)) 

if 'seq_length' in linkedin_final_df.columns:
    print(linkedin_final_df['seq_length'].max(skipna=True))

twitter_final_df.to_csv(os.path.join(save_dir, 'train_prefinal.csv'), index=False)
linkedin_final_df.to_csv(os.path.join(save_dir, 'test_prefinal.csv'), index=False)

twitter_train_data, twitter_val__data = train_test_split(twitter_final_df, test_size=0.2, random_state=42)


twitter_train_X = twitter_train_data.iloc[:, :-2]  
twitter_train_X['seq_length'] = twitter_train_data['seq_length'] 
twitter_train_y = twitter_train_data['sentiment'] 

twitter_val_X = twitter_val__data.iloc[:, :-2]
twitter_val_X['seq_length'] = twitter_val__data['seq_length'] 
twitter_val_y = twitter_val__data['sentiment'] 

linkedin_X = linkedin_final_df.iloc[:, :-2] 
linkedin_X['seq_length'] = linkedin_final_df['seq_length'] 
linkedin_y = linkedin_final_df['sentiment']  

train_csv_path = os.path.join(save_dir, 'train.csv')
test_csv_path = os.path.join(save_dir, 'test.csv')
val_csv_path = os.path.join(save_dir, 'validate.csv')
train_f_csv_path = os.path.join(save_dir, 'train_features.csv')
test_f_csv_path = os.path.join(save_dir, 'test_features.csv')
val_f_csv_path = os.path.join(save_dir, 'validate_features.csv')


twitter_train_X.to_csv(train_f_csv_path, index=False)
twitter_train_y.to_csv(train_csv_path, index=False)
twitter_val_X.to_csv(val_f_csv_path, index=False)
twitter_val_y.to_csv(val_csv_path, index=False)
linkedin_X.to_csv(test_f_csv_path, index=False)
linkedin_y.to_csv(test_csv_path, index=False)

print(f"Training set saved to {train_csv_path}")
print(f"Test set saved to {test_csv_path}")
print(f"val set saved to {val_csv_path}")

