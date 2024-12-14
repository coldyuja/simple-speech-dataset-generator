# simple-voice-dataset-generator

## Under construction.

## Prerequisites
To extract text, follow setup at https://github.com/openai/whisper

### Libraries
remove 'sudo' if needed.
```
sudo apt update && sudo apt upgrade
sudo apt install ffmpeg 
```

### Conda Environment
```
conda create --name <env> --file requirements.txt
```

## TODO List
- [ ] Make inferencing run on distributed gpu
- [ ] Automatic speaker annotations for known who already tagged by user
- [ ] Model(Detection, Separation, etc..) Training 

## License
Basically, This project is under MIT.
However, in some implementation, license may be restricted.

### Restricted Implementations
| Part                  | License       |
| :-----------          | :-----------: |
| SepReformer (Modified)| Apache 2.0    |
| Whisper               | MIT           |

### Specific Paths
| Part          | Paths                                             |
| :------------ | :---------------------------                      |
| SepReformer   | simple_vg/speech_separation/sepreformer.py        |
|               | simple_vg/speech_separation/models/SepReformer*   |
| Whisper       | simple_vg/stt/whisper_wrapper.py                  |