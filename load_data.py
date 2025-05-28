from datasets import load_dataset
import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

def load_models():
    print("SimCSE 모델을 로드합니다...")
    return SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')

def create_embeddings(abstracts, model):
    print("임베딩을 생성합니다...")
    embeddings = model.encode(abstracts, show_progress_bar=True)
    return embeddings

def load_arxiv_data():
    # 저장된 데이터 파일 경로
    cache_file = 'arxiv_abstracts.pkl'
    embeddings_file = 'arxiv_embeddings.npy'
    
    # 캐시된 파일이 있으면 로드
    if os.path.exists(cache_file) and os.path.exists(embeddings_file):
        print("캐시된 데이터를 로드합니다...")
        with open(cache_file, 'rb') as f:
            abstracts = pickle.load(f)
        embeddings = np.load(embeddings_file)
        return abstracts, embeddings
    
    # 캐시된 파일이 없으면 데이터셋 다운로드
    print("데이터셋을 다운로드합니다...")
    dataset = load_dataset("gfissore/arxiv-abstracts-2021")
    abstracts = dataset['train']['abstract']
    
    # 데이터 저장
    print("데이터를 로컬에 저장합니다...")
    with open(cache_file, 'wb') as f:
        pickle.dump(abstracts, f)
    
    # 임베딩 생성 및 저장
    model = load_models()
    embeddings = create_embeddings(abstracts, model)
    np.save(embeddings_file, embeddings)
    
    return abstracts, embeddings

