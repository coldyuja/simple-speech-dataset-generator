# simple-voice-dataset-generator

## 기말과제 제출용 Branch

## 준비사항
Whisper을 사용하기 위해 따로 https://github.com/openai/whisper 에 있는 준비사항을 따르십시오.

### 패키지 설치
```
sudo apt update && sudo apt upgrade
sudo apt install ffmpeg 
```

### Conda 환경 설정
```
conda create --name <env> --file requirements.txt
```


## 라이센스
기본적으로, 이 프로젝트는 MIT 라이센스를 따릅니다.
하지만 일부 구현에 대해서는 제한된 라이센스가 적용됩니다.

### 제한된 구현들
| Part                  | License       |
| :-----------          | :-----------: |
| SepReformer (Modified)| Apache 2.0    |
| Whisper               | MIT           |

### 상세 경로
| Part          | Paths                                             |
| :------------ | :---------------------------                      |
| SepReformer   | simple_vg/speech_separation/sepreformer.py        |
|               | simple_vg/speech_separation/models/SepReformer*   |
| Whisper       | simple_vg/stt/whisper_wrapper.py                  |