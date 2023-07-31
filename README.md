# KU-pytorch-study-2023

먼저 `ion_data/` 경로의 `train.zip`, `valid.zip` 파일의 압축을 풉니다.
```powershell
unzip ion_data/train.zip
unzip ion_data/valid.zip
```

모델 학습은 아래 코드를 실행시킵니다.
```python3
python trainer.py
```

모델 평가는 아래 코드를 실행시킵니다.
```python3
python tester.py
```