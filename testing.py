"""Verifica i tre requisiti del Project #9 eseguendoli end-to-end."""
import os
import sys
from pathlib import Path

PROGETTO = Path(__file__).resolve().parent
sys.path.insert(0, str(PROGETTO))
os.chdir(PROGETTO)

import numpy as np

from data_loader import load_behaviors, load_news
from wals import WALSModel

modello, mappings = WALSModel.load("artifacts/wmf")
news = load_news("data/train/news.tsv").set_index("news_id")
titolo = news["title"].to_dict()
categoria = news["category"].to_dict()

print("REQUISITO 1 — embedding di utenti e item")
print(f"  U: {modello.U.shape}   V: {modello.V.shape}   k = {modello.cfg.k}")
print(f"  norma media di v_j: {np.linalg.norm(modello.V, axis=1).mean():.4f}")

# REQUISITO 2: l'input e' un insieme di documenti graditi, espressi come news_id.
dev = load_behaviors("data/dev/behaviors.tsv")
esempi = []
for row in dev.itertuples(index=False):
    noti = [n for n in row.history if n in mappings.news2idx]
    if 6 <= len(noti) <= 10:
        cat = {categoria.get(n) for n in noti}
        if len(cat) <= 2:                     # profili a tema, piu' leggibili
            esempi.append((row.user_id, noti))
    if len(esempi) >= 3:
        break

print("\nREQUISITO 2 e 3 — da un insieme di documenti graditi a un ranking\n")
for user_id, graditi in esempi:
    print(f"utente {user_id}  ({len(graditi)} documenti graditi)")
    for n in graditi[:5]:
        print(f"    [{categoria.get(n,'?'):12s}] {titolo.get(n,'?')[:68]}")
    if len(graditi) > 5:
        print(f"    ... e altri {len(graditi)-5}")

    indici = np.array([mappings.news2idx[n] for n in graditi], dtype=np.int32)
    top, punteggi = modello.raccomanda(indici, top_n=5)

    print("  -> ranking restituito:")
    for r, (i, s) in enumerate(zip(top, punteggi), 1):
        nid = mappings.idx2news[i]
        print(f"    {r}. [{categoria.get(nid,'?'):12s}] {titolo.get(nid,'?')[:62]}  ({s:.3f})")
    print()

# Controllo di diversita': il modello personalizza o dice sempre le stesse cose?
campione = []
for row in dev.itertuples(index=False):
    noti = [mappings.news2idx[n] for n in row.history if n in mappings.news2idx]
    if noti:
        campione.append(np.array(noti, dtype=np.int32))
    if len(campione) >= 300:
        break

primi = [int(modello.raccomanda(h, top_n=10)[0][0]) for h in campione]
insieme_top10 = set()
for h in campione:
    insieme_top10.update(modello.raccomanda(h, top_n=10)[0].tolist())

print(f"diversita' su {len(campione)} utenti:")
print(f"  articoli distinti al primo posto : {len(set(primi))}")
print(f"  articoli distinti nelle top-10   : {len(insieme_top10)}")