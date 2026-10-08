"""Schema delle colonne dei file CSV (nessuna riga di intestazione, 19 colonne).

Semantica identificata con analisi cinematica (le posizioni sono state
derivate nel tempo e confrontate con le colonne; vedi README.md per i dettagli):

    col 0  icao24      str   indirizzo ICAO 24-bit (esadecimale)
    col 1  lat         deg   latitudine (bounding box Italia + Paesi alpini: ~39.3-51.7 N)
    col 2  lon         deg   longitudine (~2.5-15.9 E)
    col 3  track       deg   rotta vera (true track), 0=Nord, oraria  [verificato]
    col 4  alt         ft    quota barometrica in piedi
    col 5  gs          kt    velocita' al suolo in nodi             [verificato]
    col 6  -           -     colonna sempre vuota
    col 7  sector      str   settore ATC corrente (es. "F-LIRF4"); 4176 valori,
                             cambia ~ogni 30-60 s di volo. NON usato come feature.
    col 8  ac_type     str   tipo ICAO velivolo (es. A320, B738)
    col 9  reg         str   marche di registrazione
    col 10 ts          s     timestamp Unix (secondi, UTC)
    col 11 dep         str   aeroporto di partenza (IATA, es. FRA)
    col 12 arr         str   aeroporto di destinazione (IATA)
    col 13 flight_no   str   numero di volo commerciale (es. LH435)
    col 14 on_ground   0/1   1 = a terra                                     [verificato]
    col 15 vrate       ft/min rateo verticale, quantizzato a multipli di 64  [verificato]
    col 16 callsign    str   callsign radio (es. NJE325Q)
    col 17 -           0     colonna sempre 0
    col 18 operator    str   codice ICAO compagnia (es. DLH)

Nota: l'analisi ha confermato che col 3 e' la rotta (0-360), col 4 la quota
(ft), col 5 la velocita' al suolo (kt, mediana ~400 kt), col 14 il flag
"a terra" (quota 0 nel 99.7% dei casi), col 15 il rateo verticale
(correlazione 0.94 con la derivata della quota, 100% multipli di 64 ft/min).
"""

# --- colonne usate dalla pipeline (per leggere solo il necessario) -----------
USECOLS = [0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 14, 15, 18]

# --- dtype per pandas.read_csv ----------------------------------------------
DTYPES = {
    0: "category",   # icao24
    1: "float32",    # lat
    2: "float32",    # lon
    3: "float32",    # track
    4: "float32",    # alt ft
    5: "float32",    # gs kt
    8: "category",   # ac_type
    10: "int64",     # ts
    11: "category",  # dep
    12: "category",  # arr
    14: "int8",      # on_ground
    15: "float32",   # vrate ft/min
    18: "category",  # operator
}

# --- nomi interni nell'ordine di USECOLS/DTYPES ------------------------------
NEW_NAMES = [
    "icao24", "lat", "lon", "trk", "alt", "gs",
    "ac_type", "ts", "dep", "arr", "ground", "vrate", "operator",
]

# --- conversione unita' -------------------------------------------------------
KT_TO_MS = 0.514444
FT_TO_M = 0.3048
FPM_TO_MS = 0.00508
