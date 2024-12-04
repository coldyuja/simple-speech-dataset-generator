from commons import AbstractPipelineElement

# https://www.isca-archive.org/interspeech_2023/yu23c_interspeech.pdf
# Detect Overlapped Speech
# Target Speech/Speaker Extraction => SpeakerBeam, but has multihead for separation for each speaker
# OSD(Overlapped Speech Detection) => ?, 
# Not Implemented Yet 
#
# I took https://github.com/desh2608/gss but it requires multichannel source or RTTM file which must contain timestamps for all speaker
# So, i tought that it cannot be used for general purpose since it requires human-tagging task(RTTM)


class ProcessOverlappedAudio(AbstractPipelineElement):
    def __init__(self):
        
        return

    def _detect_overlapped(self):
        return
    
    def _separate_overlapped_audio(self):
        return
    

    def _process_input(self, input):
        return super()._process_input(input)
    
    def _execute(self):
        return super()._execute()
    
    def get_result(self):
        return super().get_result()


