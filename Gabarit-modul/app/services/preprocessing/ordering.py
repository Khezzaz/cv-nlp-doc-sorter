from typing import List, Dict

def order_blocks_spatial(blocks: List[Dict]) -> List[Dict]:
    """
    Sort blocks top->bottom, left->right using normalized coordinates.
    Use y_min then x_min.
    """
    def key_fn(b):
        x0, y0, x1, y1 = b["bbox"]
        return (round(y0, 4), round(x0, 4))
    return sorted(blocks, key=key_fn)
