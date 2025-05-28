import torch
import numpy as np
from load_data import load_arxiv_data
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from sentence_transformers import SentenceTransformer
import pickle
import os


def setup_gpu():
    if torch.cuda.is_available():
        print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        # GPU 메모리 캐시 초기화
        torch.cuda.empty_cache()
        return 'cuda'
    else:
        print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")
        return 'cpu'

def load_models():
    print("모델을 로드합니다...")
    device = setup_gpu()
    
    # PEGASUS 모델 로드
    pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
    pegasus_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
    pegasus_model = pegasus_model.to(device)
    
    # SimCSE 모델 로드
    simcse_model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased')
    if device == 'cuda':
        simcse_model = simcse_model.to(device)
    
    return pegasus_tokenizer, pegasus_model, simcse_model

def generate_summaries(abstracts, tokenizer, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("초록을 요약합니다...")
    model.to(device)
    model.eval()  # 평가 모드로 설정
    
    summaries = []
    batch_size = 8  # GPU 메모리에 맞게 배치 크기 증가
    
    with torch.no_grad():
        for i in range(0, len(abstracts), batch_size):
            batch_abstracts = abstracts[i:i+batch_size]
            
            # 토큰화
            inputs = tokenizer(batch_abstracts, truncation=True, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 요약 생성
            summary_ids = model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            # 디코딩
            batch_summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]
            summaries.extend(batch_summaries)
            
            if i % 100 == 0:
                print(f"진행률: {i}/{len(abstracts)} 초록 요약 완료")
                # GPU 메모리 정리
                if device == 'cuda':
                    torch.cuda.empty_cache()
    
    return summaries

def create_embeddings(texts, model):
    print("임베딩을 생성합니다...")
    return model.encode(texts, show_progress_bar=True)

def get_similar_papers(query, summaries, embeddings, simcse_model, top_n=5):
    print(f"'{query}'에 대한 유사 논문을 검색합니다...")
    # 쿼리 임베딩 생성
    query_embedding = simcse_model.encode([query])
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # 상위 N개 인덱스 찾기
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    return [(summaries[idx], similarities[idx]) for idx in top_indices]

def load_or_create_summaries_and_embeddings(abstracts, pegasus_tokenizer, pegasus_model, simcse_model):
    # 저장된 파일 경로
    summaries_file = 'arxiv_summaries.pkl'
    embeddings_file = 'arxiv_summaries_embeddings.npy'
    
    # 저장된 파일이 있으면 로드
    if os.path.exists(summaries_file) and os.path.exists(embeddings_file):
        print("저장된 요약문과 임베딩을 로드합니다...")
        with open(summaries_file, 'rb') as f:
            summaries = pickle.load(f)
        embeddings = np.load(embeddings_file)
        return summaries, embeddings
    
    # 없으면 새로 생성
    print("요약문과 임베딩을 새로 생성합니다...")
    summaries = generate_summaries(abstracts, pegasus_tokenizer, pegasus_model)
    embeddings = create_embeddings(summaries, simcse_model)
    
    # 저장
    print("요약문과 임베딩을 저장합니다...")
    with open(summaries_file, 'wb') as f:
        pickle.dump(summaries, f)
    np.save(embeddings_file, embeddings)
    
    return summaries, embeddings

def main():
    # 데이터 로드
    print("데이터를 로드합니다...")
    abstracts = load_arxiv_data()
    print(f"로드된 초록 수: {len(abstracts)}")
    
    # 모델 로드
    pegasus_tokenizer, pegasus_model, simcse_model = load_models()
    
    # 요약문과 임베딩 로드 또는 생성
    summaries, embeddings = load_or_create_summaries_and_embeddings(
        abstracts, pegasus_tokenizer, pegasus_model, simcse_model
    )
    
    # 검색 예시
    while True:
        query = input("\n검색할 쿼리를 입력하세요 (종료하려면 'q' 입력): ")
        if query.lower() == 'q':
            break
            
        similar_papers = get_similar_papers(query, summaries, embeddings, simcse_model)
        
        print("\n상위 5개 유사 논문:")
        for i, (summary, similarity) in enumerate(similar_papers, 1):
            print(f"\n{i}위 (유사도: {similarity:.4f})")
            print(f"요약: {summary}")

if __name__ == "__main__":
    main()