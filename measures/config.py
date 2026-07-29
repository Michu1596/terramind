import numpy as np

TRANSFORMATIONS = {
    "L": {
        "pattern": "L_geom_seed*.csv",
        "meta_idx": 0,
        "circular": False,
        "time_filter": False,
        "name": "Latitude",
    },
    "Long": {
        "pattern": "Long_geom_seed*.csv",
        "meta_idx": 1,
        "circular": True,
        "time_filter": False,
        "name": "Longitude",
        "transform": lambda x: np.pi * (x + 180) / 180,
    },
    "H": {
        "pattern": "H_geom_seed*.csv",
        "meta_idx": 3,
        "circular": False,
        "time_filter": False,
        "name": "Altitude",
    },
    "T": {
        "pattern": "T_geom_seed*.csv",
        "meta_idx": -1,
        "circular": True,
        "time_filter": True,
        "name": "Year progression",
    },
}
