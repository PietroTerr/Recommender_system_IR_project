"""Confronto fra la Weighted MF e le baseline sul dev set.

Tutti i modelli sono valutati sulla stessa struttura di valutazione e con gli
stessi due protocolli:

- per impression, riordinando i ~37 candidati mostrati da MSN (protocollo
  ufficiale di MIND);
- full-catalog, scegliendo fra tutti gli articoli del training ed escludendo
  quelli gia' letti. E' il compito che il testo del progetto descrive.

Oltre ai modelli singoli viene valutato l'ibrido WMF + CTR: serve a stabilire
se il segnale collaborativo della fattorizzazione aggiunge qualcosa alla
popolarita' corretta per esposizione, o se ne e' solo una copia peggiore.
"""

import numpy as np

from data_loader import load_artifacts
from evaluation import (GlobalScorer, RandomScorer, Scorer, ScorerIbrido,
                        evaluate, evaluate_recall_full_catalog,
                        grafico_11_punti, load_eval_set,
                        popularity_from_feedback, smoothed_ctr, stampa_tabella)
from wals import WALSModel


class ScorerWMF(Scorer):
    """La Weighted MF nell'interfaccia di valutazione.

    Il vettore utente e' sempre ricavato dal profilo con il fold-in: sul dev
    l'88% degli utenti non ha una riga in U, e per gli altri si e' verificato
    che le due strade danno lo stesso risultato (MAP 0,2618 contro 0,2632).
    Gli articoli senza colonna in V ricevono u . media(V), cioe' il punteggio
    che l'utente darebbe a un articolo medio del catalogo.
    """

    def __init__(self, modello: WALSModel, name="WMF"):
        self.m = modello
        self.media_item = modello.V.mean(axis=0)
        self.name = name

    def _u(self, user_idx, history):
        if len(history):
            return self.m.fold_in(history)
        return self.m.U[user_idx] if user_idx >= 0 else None

    def scores_for(self, user_idx, history, item_idx):
        u = self._u(user_idx, history)
        if u is None:
            return np.zeros(len(item_idx))
        noto = item_idx >= 0
        safe = np.maximum(item_idx, 0)
        return np.where(noto, self.m.V[safe] @ u, float(u @ self.media_item))

    def all_item_scores(self, user_idx, history):
        u = self._u(user_idx, history)
        return np.zeros(len(self.m.V)) if u is None else self.m.V @ u


def main():
    art = load_artifacts("artifacts/train")
    mappings, C_pos = art["mappings"], art["C_pos"]
    eval_set = load_eval_set("artifacts/dev/eval_set.npz")
    print(f"impression valutate: {len(eval_set):,}   "
          f"candidati senza colonna in V: {eval_set.stats['quota_cold']:.1%}\n")

    with np.load("artifacts/train/item_stats.npz") as d:
        ctr = GlobalScorer(smoothed_ctr(d["clicks"], d["shows"]), name="CTR smussato")

    modello, _ = WALSModel.load("artifacts/wmf")
    wmf = ScorerWMF(modello)

    scorers = [
        RandomScorer(n_items=len(mappings.idx2news), seed=0),
        GlobalScorer(popularity_from_feedback(C_pos), name="popolarita"),
        ctr,
        wmf,
        ScorerIbrido(wmf, ctr, name="ibrido (WMF + CTR)"),
    ]

    righe, curve = [], {}
    for s in scorers:
        risultati, curva = evaluate(s, eval_set)
        righe.append(risultati)
        curve[s.name] = curva
        print(f"  {s.name:22s} MAP {risultati['MAP']:.4f}")

    print("\nProtocollo per impression")
    stampa_tabella(righe)
    grafico_11_punti(curve, "artifacts/confronto_11_punti.png")

    print("\nProtocollo full-catalog (campione di 2000 utenti)")
    stampa_tabella([evaluate_recall_full_catalog(s, eval_set, max_users=2000) for s in scorers])


if __name__ == "__main__":
    main()
