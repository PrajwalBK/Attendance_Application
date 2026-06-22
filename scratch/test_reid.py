import sys
try:
    import torchreid
    print("torchreid found")
    print("model_urls:", list(torchreid.models.osnet.model_urls.keys()))
except Exception as e:
    print("Error:", e)
