from typing import List, Dict

def filter_blocks(blocks: List[Dict], min_area: float = 0.0005, min_confidence: float = 0.4) -> List[Dict]:
    filtered = []
    for b in blocks:
        if b.get("area", 0.0) < min_area:
            continue
        if b.get("confidence", 0.0) < min_confidence:
            continue
        filtered.append(b)
    return filtered
