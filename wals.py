"""Weighted Matrix Factorisation risolta con Weighted Alternating Least Squares.

Funzione obiettivo (lezione 9). Con C la matrice di feedback, U (utenti x k) e
V (item x k) i fattori, e Ĉ = U Vᵀ l'approssimazione:

    min  Σ  w_ij (C_ij - u_i·v_j)²  +  w₀ Σ (u_i·v_j)²  +  λ(‖U‖² + ‖V‖²)
        i,j ∈ Oss                        i,j ∉ Oss

Il dataset MIND consente di distinguere *tre* casi invece dei soliti due,
perche' gli impression log dicono anche cosa e' stato mostrato senza essere
cliccato:

    positivo      (i,j) ∈ P    C_ij = 1   peso w_pos
    negativo      (i,j) ∈ N    C_ij = 0   peso w_neg     mostrato, non cliccato
    non osservato              C_ij = 0   peso w₀        mai mostrato

Il caso "observed-only" si ottiene con w₀ = 0, quello della SVD con
w₀ = w_pos = w_neg: la parametrizzazione include entrambi gli estremi
discussi a lezione.

Il trucco che rende praticabile l'algoritmo. Annullando il gradiente rispetto
a u_i si ottiene un sistema lineare k x k la cui matrice, scritta in modo
ingenuo, e' una somma su tutte le 51.282 colonne. Poiche' pero' tutte le celle
non osservate hanno lo stesso peso w₀, la somma si spezza in un termine
uguale per ogni utente piu' una correzione sulle sole celle osservate:

    A_i = w₀ VᵀV + (w_pos - w₀) Σ v_j v_jᵀ + (w_neg - w₀) Σ v_j v_jᵀ + λI
                                j ∈ P_i                    j ∈ N_i
    b_i = w_pos  Σ  v_j                    e infine   u_i = A_i⁻¹ b_i
                j ∈ P_i

La Gramiana VᵀV e' k x k e si calcola una volta per mezza iterazione; la
correzione tocca solo le ~23 celle positive e ~96 negative dell'utente. Il
costo per utente passa cosi' da O(n·k²) a O((|P_i|+|N_i|)·k² + k³). Il ramo
simmetrico fissa U e risolve per v_j, e richiede l'accesso *per colonna* alle
matrici sparse: per questo si tengono pronte sia la vista CSR sia la CSC.
"""

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from data_loader import load_artifacts, save_artifacts

# Le predizioni sulle celle osservate si calcolano a blocchi: U[righe] su 6
# milioni di celle in una volta sola occuperebbe qualche gigabyte.
BLOCCO = 250_000


@dataclass
class WALSConfig:
    k: int = 32              # numero di fattori latenti
    w_pos: float = 1.0       # peso delle celle positive (riferimento)
    w_neg: float = 0.10      # peso di "mostrato ma non cliccato"
    w0: float = 0.025        # peso delle celle mai osservate
    reg: float = 0.05        # coefficiente di regolarizzazione λ
    n_iter: int = 15
    seed: int = 0


# --- Passo di alternanza -------------------------------------------------

def _risolvi_fattori(F, indptr_p, indices_p, indptr_n, indices_n, n, cfg) -> np.ndarray:
    """Risolve un lato dell'alternanza tenendo fissa la matrice F.

    Vale sia per gli utenti (F = V, una riga per utente) sia per gli item
    (F = U, una riga per item): e' lo stesso sistema lineare, cambia solo su
    quale indice si itera e se le matrici sparse sono lette per riga o per
    colonna.
    """
    k = cfg.k
    # Termine comune a tutte le entita': si calcola una volta sola.
    G = cfg.w0 * (F.T @ F) + cfg.reg * np.eye(k)
    corr_pos = cfg.w_pos - cfg.w0
    corr_neg = cfg.w_neg - cfg.w0

    X = np.zeros((n, k))
    for i in range(n):
        p = indices_p[indptr_p[i]:indptr_p[i + 1]]
        q = indices_n[indptr_n[i]:indptr_n[i + 1]]

        A = G.copy()
        if len(p):
            Fp = F[p]
            A += corr_pos * (Fp.T @ Fp)
            b = cfg.w_pos * Fp.sum(axis=0)
        else:
            # Nessun positivo: il termine noto e' nullo e la soluzione e' 0.
            # Capita per gli item mai cliccati da nessuno.
            b = np.zeros(k)
        if len(q):
            Fq = F[q]
            A += corr_neg * (Fq.T @ Fq)

        X[i] = np.linalg.solve(A, b)
    return X


def _predizioni(U, V, righe, colonne) -> np.ndarray:
    """Valori di u_i·v_j sulle sole celle indicate, calcolati a blocchi."""
    out = np.empty(len(righe))
    for inizio in range(0, len(righe), BLOCCO):
        sl = slice(inizio, min(inizio + BLOCCO, len(righe)))
        out[sl] = np.einsum("ij,ij->i", U[righe[sl]], V[colonne[sl]], optimize=True)
    return out


def calcola_perdita(U, V, celle_pos, celle_neg, cfg) -> float:
    """Valore esatto della funzione obiettivo, senza espandere la matrice densa.

    Il termine sulle celle non osservate richiederebbe una somma su tutte le
    50.000 x 51.282 celle. Si evita osservando che

        Σ (u_i·v_j)² = traccia( (UᵀU)(VᵀV) )
       i,j

    che costa O((m+n)k² + k³): si calcola la somma su *tutte* le celle e si
    sottraggono quelle osservate, che sono note una per una.
    """
    y_pos = _predizioni(U, V, *celle_pos)
    y_neg = _predizioni(U, V, *celle_neg)

    somma_totale = float(np.sum((U.T @ U) * (V.T @ V)))
    somma_osservate = float(np.sum(y_pos ** 2) + np.sum(y_neg ** 2))

    return (
        cfg.w_pos * float(np.sum((1.0 - y_pos) ** 2))
        + cfg.w_neg * float(np.sum(y_neg ** 2))
        + cfg.w0 * (somma_totale - somma_osservate)
        + cfg.reg * (float(np.sum(U * U)) + float(np.sum(V * V)))
    )


# --- Modello -------------------------------------------------------------

class WALSModel:
    """Fattori U e V, con il fold-in di un utente nuovo e la raccomandazione."""

    def __init__(self, U: np.ndarray, V: np.ndarray, cfg: WALSConfig):
        self.U = U
        self.V = V
        self.cfg = cfg
        # Sistema del fold-in: la parte che non dipende dall'utente.
        self._G = cfg.w0 * (V.T @ V) + cfg.reg * np.eye(cfg.k)
        self._cache = {}

    def fold_in(self, history: np.ndarray) -> np.ndarray:
        """Vettore latente di un utente descritto dai soli documenti graditi.

        E' esattamente un passo del ramo "fissa V e risolvi per u", ristretto
        alle celle positive: di un utente nuovo non si sa cosa gli sia stato
        mostrato senza essere cliccato, quindi il termine sui negativi non
        c'e'. E' l'input richiesto dal progetto ("a set of liked documents"),
        e serve per l'88% delle impression del dev, i cui utenti non compaiono
        nel training.

        I risultati sono memorizzati: nel dev lo stesso utente ricorre in piu'
        impression con la stessa History, e risolvere una volta basta.
        """
        chiave = history.tobytes()
        u = self._cache.get(chiave)
        if u is None:
            Vh = self.V[history]
            A = self._G + (self.cfg.w_pos - self.cfg.w0) * (Vh.T @ Vh)
            u = np.linalg.solve(A, self.cfg.w_pos * Vh.sum(axis=0))
            self._cache[chiave] = u
        return u

    def raccomanda(self, history: np.ndarray, top_n: int = 10) -> tuple:
        """Ranking dei documenti per un utente dato il suo insieme di gradimenti.

        Restituisce (indici, punteggi) dei top_n item, esclusi quelli gia'
        presenti nel profilo: raccomandare un articolo gia' letto non serve.
        """
        u = self.fold_in(history)
        punteggi = self.V @ u
        punteggi[history] = -np.inf
        top = np.argpartition(-punteggi, top_n)[:top_n]
        top = top[np.argsort(-punteggi[top], kind="stable")]
        return top, punteggi[top]

    def save(self, out_dir: str, mappings) -> None:
        save_artifacts(out_dir, mappings, U=self.U, V=self.V)
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)

    @staticmethod
    def load(out_dir: str) -> tuple:
        """Ricarica il modello addestrato: (modello, mappature)."""
        art = load_artifacts(out_dir)
        with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
            cfg = WALSConfig(**json.load(f))
        return WALSModel(art["U"], art["V"], cfg), art["mappings"]


# --- Addestramento -------------------------------------------------------

def train(C_pos, M_neg, cfg: WALSConfig, verbose: bool = True) -> tuple:
    """Esegue WALS e restituisce (modello, storia della perdita).

    L'alternanza e' garantita non crescente: ogni mezzo passo risolve
    *esattamente* il minimo rispetto a un blocco di variabili tenendo fisso
    l'altro. La perdita stampata a ogni iterazione serve proprio a
    verificarlo: se sale, c'e' un errore nelle equazioni normali.
    """
    n_utenti, n_item = C_pos.shape
    rng = np.random.default_rng(cfg.seed)

    # Vista per riga (ramo utenti) e per colonna (ramo item).
    pos_r, neg_r = C_pos.tocsr(), M_neg.tocsr()
    pos_c, neg_c = C_pos.tocsc(), M_neg.tocsc()

    # Celle osservate in forma di coordinate, per il calcolo della perdita.
    coo_pos, coo_neg = C_pos.tocoo(), M_neg.tocoo()
    celle_pos = (coo_pos.row, coo_pos.col)
    celle_neg = (coo_neg.row, coo_neg.col)

    # V va inizializzata a caso, U verra' sovrascritta al primo mezzo passo.
    U = np.zeros((n_utenti, cfg.k))
    V = rng.normal(0.0, 1.0 / np.sqrt(cfg.k), size=(n_item, cfg.k))

    storia = []
    for it in range(1, cfg.n_iter + 1):
        t0 = time.perf_counter()

        U = _risolvi_fattori(V, pos_r.indptr, pos_r.indices,
                             neg_r.indptr, neg_r.indices, n_utenti, cfg)
        V = _risolvi_fattori(U, pos_c.indptr, pos_c.indices,
                             neg_c.indptr, neg_c.indices, n_item, cfg)

        perdita = calcola_perdita(U, V, celle_pos, celle_neg, cfg)
        storia.append(perdita)
        if verbose:
            variazione = "" if it == 1 else f"  ({perdita - storia[-2]:+.1f})"
            print(f"  iterazione {it:2d}  perdita {perdita:14,.1f}{variazione}"
                  f"   {time.perf_counter() - t0:5.1f}s")

    return WALSModel(U, V, cfg), storia


def grafico_convergenza(storia, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(storia) + 1), storia, marker="o", markersize=4)
    ax.set_xlabel("Iterazione")
    ax.set_ylabel("Funzione obiettivo")
    ax.set_title("Convergenza di WALS")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from evaluation import (GlobalScorer, MFScorer, evaluate,
                            evaluate_recall_full_catalog, grafico_11_punti,
                            load_eval_set, smoothed_ctr, stampa_tabella)

    ARTIFACTS = "artifacts/train"
    MODELLO = "artifacts/wmf"

    art = load_artifacts(ARTIFACTS)
    mappings, C_pos, M_neg = art["mappings"], art["C_pos"], art["M_neg"]

    cfg = WALSConfig()
    print(f"WALS  k={cfg.k}  w_pos={cfg.w_pos}  w_neg={cfg.w_neg}  "
          f"w0={cfg.w0}  reg={cfg.reg}  iterazioni={cfg.n_iter}")
    print(f"C: {C_pos.shape[0]:,} utenti x {C_pos.shape[1]:,} item   "
          f"{C_pos.nnz:,} positivi   {M_neg.nnz:,} negativi osservati\n")

    modello, storia = train(C_pos, M_neg, cfg)
    modello.save(MODELLO, mappings)
    grafico_convergenza(storia, "artifacts/wmf/convergenza.png")

    monotona = all(b <= a + 1e-6 for a, b in zip(storia, storia[1:]))
    print(f"\nPerdita non crescente a ogni iterazione: {monotona}")
    print(f"Modello salvato in {MODELLO}")

    # --- Valutazione sul dev, contro la migliore baseline -----------------
    eval_set = load_eval_set("artifacts/dev/eval_set.npz")
    with np.load("artifacts/train/item_stats.npz") as d:
        ctr = GlobalScorer(smoothed_ctr(d["clicks"], d["shows"]), name="CTR smussato")

    scorers = [
        ctr,
        MFScorer(modello.U, modello.V, fold_in=modello.fold_in,
                 fallback_scorer=ctr, name=f"WMF k={cfg.k}"),
    ]

    righe, curve = [], {}
    for s in scorers:
        risultati, curva = evaluate(s, eval_set)
        righe.append(risultati)
        curve[s.name] = curva

    print("\nProtocollo per impression")
    stampa_tabella(righe)
    grafico_11_punti(curve, "artifacts/wmf/curva_11_punti.png")

    print("\nProtocollo full-catalog (campione di 2000 utenti)")
    stampa_tabella([evaluate_recall_full_catalog(s, eval_set, max_users=2000) for s in scorers])
