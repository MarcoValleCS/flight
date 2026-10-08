"""FlightPREV - Previsione a breve termine di traiettorie aeree.

Pipeline:
    1. src.build_dataset  : CSV OpenFlight -> array a finestre (sliding windows)
    2. src.train          : addestramento encoder Transformer/GRU (PyTorch)
    3. src.evaluate       : metriche, confronto con baseline cinematiche, grafici
    4. src.predict        : inferenza su una traccia nuova
"""

__version__ = "1.0.0"
