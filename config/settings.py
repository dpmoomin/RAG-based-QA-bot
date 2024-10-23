# config/settings.py
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 가져오기
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")