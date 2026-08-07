"""Demo da riga di comando: da un insieme di documenti graditi a un ranking.

E' il requisito del progetto reso eseguibile: "The system must accept as input
'a user' (a set of liked documents)" e "Return a ranking of documents".

    python recommend.py --news N55189 N42782 N34694
    python recommend.py --utente U38418
    python recommend.py                       # esempio a caso dal dev

Il vettore utente e' sempre ricavato dal profilo con il fold-in, anche quando
l'utente esiste nel training: e' un singolo passo di WALS a V fissata, e sul
dev di MIND e' l'unica strada possibile, dato che l'88% degli utenti non
compare fra le righe di U.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

from data_loader import load_index, load_news

RADICE = Path(__file__).resolve().parent
BEHAVIORS_DEV = RADICE / "data/dev/behaviors.tsv"


def carica_modello(cartella: Path):
    """Carica i fattori U e V salvati dal training e ricostruisce il modello."""
    from wals import WALSModel
    modello, _ = WALSModel.load(str(cartella))
    return modello


def history_di(user_id: str):
    """Cerca la History di un utente scorrendo behaviors.tsv.

    Si legge riga per riga con uscita anticipata invece di caricare i 40 MB in
    un DataFrame: qui serve un solo utente, e la demo deve rispondere subito.
    """
    if not BEHAVIORS_DEV.exists():
        sys.exit(f"manca {BEHAVIORS_DEV}: eseguire prima data_downloader.py")
    with open(BEHAVIORS_DEV, encoding="utf-8") as f:
        for riga in f:
            campi = riga.rstrip("\n").split("\t")
            if len(campi) >= 4 and campi[1] == user_id:
                return campi[3].split()
    return None


def utente_a_caso(rng: random.Random, min_articoli: int = 5):
    """Un utente del dev con un profilo abbastanza ricco da essere leggibile."""
    candidati = []
    with open(BEHAVIORS_DEV, encoding="utf-8") as f:
        for i, riga in enumerate(f):
            if i >= 3000:
                break
            campi = riga.rstrip("\n").split("\t")
            if len(campi) >= 4 and len(campi[3].split()) >= min_articoli:
                candidati.append((campi[1], campi[3].split()))
    return rng.choice(candidati)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    gruppo = parser.add_mutually_exclusive_group()
    gruppo.add_argument("--news", nargs="+", metavar="ID",
                        help="identificatori degli articoli graditi (es. N55189)")
    gruppo.add_argument("--utente", metavar="USER_ID",
                        help="prende il profilo dalla History di un utente del dev")
    parser.add_argument("--modello", default="artifacts/wmf",
                        help="cartella del modello (default: artifacts/wmf)")
    parser.add_argument("--top", type=int, default=10, help="quanti articoli restituire")
    parser.add_argument("--seed", type=int, default=None, help="seme per l'esempio a caso")
    args = parser.parse_args()

    cartella = RADICE / args.modello
    if not (cartella / "V.npy").exists():
        sys.exit(f"nessun modello in {cartella}: eseguire prima wals.py")

    modello = carica_modello(cartella)
    mappings = load_index(str(cartella / "index.npz"))

    # I titoli servono solo per la stampa. Il dev aggiunge gli articoli
    # comparsi dopo il training: compaiono nei profili ma non hanno una
    # colonna in V, quindi vanno mostrati e poi scartati.
    news = load_news(str(RADICE / "data/train/news.tsv"))
    dev = load_news(str(RADICE / "data/dev/news.tsv"))
    news = news.set_index("news_id")
    dev = dev.set_index("news_id")
    titolo = {**dev["title"].to_dict(), **news["title"].to_dict()}
    categoria = {**dev["category"].to_dict(), **news["category"].to_dict()}

    # --- Da cosa e' composto il profilo -----------------------------------
    if args.news:
        user_id, graditi = None, args.news
    elif args.utente:
        graditi = history_di(args.utente)
        if graditi is None:
            sys.exit(f"utente {args.utente} non trovato nel dev")
        user_id = args.utente
    else:
        user_id, graditi = utente_a_caso(random.Random(args.seed))
        print(f"(nessun argomento: uso l'utente {user_id} preso dal dev)\n")

    noti = [n for n in graditi if n in mappings.news2idx]
    ignoti = [n for n in graditi if n not in mappings.news2idx]
    if not noti:
        sys.exit("nessuno degli articoli indicati compare nel training: "
                 "senza almeno un articolo noto non e' possibile collocare l'utente.")

    # --- Stampa del profilo ----------------------------------------------
    print(f"modello: Weighted MF k={modello.cfg.k}  "
          f"({len(mappings.idx2news):,} articoli, {len(mappings.idx2user):,} utenti)")
    if user_id:
        nel_training = user_id in mappings.user2idx
        print(f"utente:  {user_id}  —  {'presente' if nel_training else 'NON presente'} "
              f"fra i {len(mappings.idx2user):,} del training")
    print(f"\nDocumenti graditi in ingresso ({len(noti)} utilizzabili"
          f"{f', {len(ignoti)} senza rappresentazione' if ignoti else ''}):")
    for n in noti:
        print(f"  [{categoria.get(n, '?'):12s}] {titolo.get(n, '(titolo non disponibile)')[:70]}")
    for n in ignoti:
        print(f"  [{'non nel modello':15s}] {titolo.get(n, n)[:67]}")

    # --- Ranking ----------------------------------------------------------
    indici = np.array([mappings.news2idx[n] for n in noti], dtype=np.int32)
    t0 = time.perf_counter()
    posizioni, punteggi = modello.raccomanda(indici, top_n=args.top)
    durata = (time.perf_counter() - t0) * 1000

    print(f"\nRanking dei documenti (primi {args.top} su {len(mappings.idx2news):,}, "
          f"esclusi quelli gia' nel profilo):")
    for r, (i, s) in enumerate(zip(posizioni, punteggi), 1):
        nid = mappings.idx2news[i]
        print(f"  {r:2d}. {s:6.3f}  [{categoria.get(nid, '?'):12s}] "
              f"{titolo.get(nid, nid)[:62]}")

    print(f"\nprodotto in {durata:.1f} ms (fold-in + punteggio su tutto il catalogo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python recommend.py --utente U13740 --top 5
# python recommend.py --news N55189 N42782 N34694
