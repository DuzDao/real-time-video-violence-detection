import assemblyai as aai

class AudioViolenceDetector:
    def __init__(self, config):
        aai.settings.api_key = config["assembly_api_key"]
        aai_conf = aai.TranscriptionConfig(content_safety=True)
        self.transcriber = aai.Transcriber(config=aai_conf)
    
    def get_hate_speech(self, mp4_path):
        """
        Get transcripts of video with path `mp4_path`.
        """
        hate_speechs = []
        print("Detecting violence transcripts...")
        transcript = self.transcriber.transcribe(mp4_path)
        for result in transcript.content_safety.results:
            for label in result.labels:
                if label.label == "profanity":
                    hate_speechs.append((result.text, label.label, label.confidence, label.severity))
        return hate_speechs
    
