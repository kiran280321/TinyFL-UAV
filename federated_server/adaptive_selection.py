import random

def select_best_uavs(uav_list, k=3):
    """Sort UAVs based on battery + signal strength."""
    ranked = sorted(
        uav_list,
        key=lambda u: (u["battery"], u["signal"]),
        reverse=True
    )
    return ranked[:k]
