# DEPRECATED: Person ReID cache manager has been completely removed per system requirement.
# This file is kept as an empty stub to prevent import errors during transition.

class ReIDCacheManager:
    def __init__(self, *args, **kwargs):
        pass
    def load_today(self):
        return {}
    def save(self, gallery):
        return {}
    def get_stats(self):
        return {"date": "", "persons": 0, "embeddings": 0, "size_kb": 0}
