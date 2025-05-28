# 기본 CUDA 이미지 사용
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# 시간대 설정
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Python 3.9 설치
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    build-essential \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 애플리케이션 파일 복사
COPY . .

# 포트 설정 (필요한 경우)
EXPOSE 8000

# 애플리케이션 실행
CMD ["python3", "main.py"]