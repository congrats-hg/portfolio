# 환경 활성화 (필수)
source /home/ryu5090/miniconda3/etc/profile.d/conda.sh && conda activate g-f-s

# 파일 업로드 (전체)
python main.py upload --resume

# 파일 업로드 (제한)
python main.py upload --limit 100

# 검색
python main.py search "검색어"

# 상태 확인
python main.py status

# 대화형 검색
python main.py interactive
