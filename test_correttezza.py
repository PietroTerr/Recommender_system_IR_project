"""Test di correttezza del sistema.

    python test_correttezza.py              # test veloci, non serve il dataset
    python test_correttezza.py --dataset    # aggiunge i controlli sugli artefatti reali

Il rischio di questo progetto non sono i crash, sono i numeri sbagliati ma
plausibili: un MAP di 0,26 sembra ragionevole sia che il codice sia giusto sia
che si stia deduplicando male il feedback o sbagliando un segno. I test qui
sotto verificano quindi le *identita' matematiche* su cui poggia
l'implementazione, confrontando ogni forma ottimizzata con la definizione
calcolata per esteso su matrici giocattolo.
"""

import argparse
import sys

import numpy as np
from scipy.sparse import csr_matrix

from evaluation import (average_precision, ordina_per_score, precision_at_k,
                        precisione_interpolata, r_precision, test_metriche)
from wals import WALSConfig, WALSModel, _risolvi_fattori, calcola_perdita, train


# --- Dati giocattolo -----------------------------------------------------

def matrici_giocattolo(n_utenti=40, n_item=30, seed=0, con_negativi=True):
    """Feedback casuale abbastanza piccolo da poter calcolare tutto per esteso.

    Positivi e negativi sono disgiunti per costruzione (partizionano una
    permutazione), come nella matrice reale dopo la precedenza ai positivi.
    """
    rng = np.random.default_rng(seed)
    rp, cp, rn, cn = [], [], [], []
    for i in range(n_utenti):
        perm = rng.permutation(n_item)
        n_p = int(rng.integers(1, 6))
        n_n = int(rng.integers(0, 9)) if con_negativi else 0
        rp += [i] * n_p
        cp += perm[:n_p].tolist()
        rn += [i] * n_n
        cn += perm[n_p:n_p + n_n].tolist()

    forma = (n_utenti, n_item)
    C_pos = csr_matrix((np.ones(len(rp), np.float32), (rp, cp)), shape=forma)
    M_neg = csr_matrix((np.ones(len(rn), np.float32), (rn, cn)), shape=forma)
    return C_pos, M_neg


# --- Implementazioni di riferimento, scritte sulla definizione ------------

def _risolvi_ingenuo(F, pos, neg, n, cfg):
    """Il passo di alternanza come lo si scriverebbe leggendo la formula.

    Costruisce il vettore dei pesi su *tutte* le colonne e somma su tutte:
    e' O(n·k²) per entita', impraticabile sui dati veri, ma qui e' proprio il
    punto — e' la verita' contro cui va confrontata la versione ottimizzata.
    """
    k = cfg.k
    X = np.zeros((n, k))
    for i in range(n):
        p = pos.indices[pos.indptr[i]:pos.indptr[i + 1]]
        q = neg.indices[neg.indptr[i]:neg.indptr[i + 1]]

        omega = np.full(F.shape[0], cfg.w0)
        omega[p] = cfg.w_pos
        omega[q] = cfg.w_neg

        A = (F * omega[:, None]).T @ F + cfg.reg * np.eye(k)
        b = cfg.w_pos * F[p].sum(axis=0) if len(p) else np.zeros(k)
        X[i] = np.linalg.solve(A, b)
    return X


def _perdita_ingenua(U, V, C_pos, M_neg, cfg):
    """La funzione obiettivo espansa sulla matrice densa, cella per cella."""
    predetto = U @ V.T
    pesi = np.full(predetto.shape, cfg.w0)
    bersaglio = np.zeros(predetto.shape)

    pos, neg = C_pos.tocoo(), M_neg.tocoo()
    pesi[pos.row, pos.col] = cfg.w_pos
    bersaglio[pos.row, pos.col] = 1.0
    pesi[neg.row, neg.col] = cfg.w_neg

    return float((pesi * (bersaglio - predetto) ** 2).sum()
                 + cfg.reg * ((U * U).sum() + (V * V).sum()))


# --- Test ----------------------------------------------------------------

def test_gramiana_ramo_utenti():
    """La riscrittura con la Gramiana deve dare gli stessi fattori della somma piena.

    E' il test piu' importante del progetto: l'intera implementazione poggia
    sull'identita'

        Σ_j ω_ij v_j v_jᵀ  =  w₀·VᵀV + Σ_{j∈P_i}(w_pos−w₀)v_jv_jᵀ
                                     + Σ_{j∈N_i}(w_neg−w₀)v_jv_jᵀ

    Se un segno fosse sbagliato il modello continuerebbe ad addestrarsi e la
    perdita a scendere: nessun altro controllo se ne accorgerebbe.
    """
    cfg = WALSConfig(k=8, w_pos=1.0, w_neg=0.37, w0=0.041, reg=0.05)
    C_pos, M_neg = matrici_giocattolo()
    rng = np.random.default_rng(1)
    V = rng.normal(size=(C_pos.shape[1], cfg.k))

    pos, neg = C_pos.tocsr(), M_neg.tocsr()
    veloce = _risolvi_fattori(V, pos.indptr, pos.indices,
                              neg.indptr, neg.indices, C_pos.shape[0], cfg)
    lento = _risolvi_ingenuo(V, pos, neg, C_pos.shape[0], cfg)

    assert np.allclose(veloce, lento, atol=1e-10), np.abs(veloce - lento).max()


def test_gramiana_ramo_item():
    """Lo stesso, sul ramo che fissa U e risolve per gli item.

    Qui le matrici sparse si leggono per colonna (CSC): il test verifica anche
    che gli indici non siano stati scambiati fra i due rami.
    """
    cfg = WALSConfig(k=6, w_pos=1.0, w_neg=0.5, w0=0.02, reg=0.1)
    C_pos, M_neg = matrici_giocattolo()
    rng = np.random.default_rng(2)
    U = rng.normal(size=(C_pos.shape[0], cfg.k))

    pos, neg = C_pos.tocsc(), M_neg.tocsc()
    veloce = _risolvi_fattori(U, pos.indptr, pos.indices,
                              neg.indptr, neg.indices, C_pos.shape[1], cfg)
    lento = _risolvi_ingenuo(U, pos, neg, C_pos.shape[1], cfg)

    assert np.allclose(veloce, lento, atol=1e-10), np.abs(veloce - lento).max()


def test_perdita_in_forma_chiusa():
    """Il termine sulle celle non osservate e' calcolato senza espanderle.

    Verifica  Σ_ij (u_i·v_j)² = traccia((UᵀU)(VᵀV))  nel contesto in cui e'
    usata: se fosse sbagliata, la perdita resterebbe monotona (perche' i
    fattori sono comunque ottimi) ma il suo valore non significherebbe nulla.
    """
    cfg = WALSConfig(k=5, w_pos=1.0, w_neg=0.3, w0=0.07, reg=0.02)
    C_pos, M_neg = matrici_giocattolo(n_utenti=25, n_item=18, seed=3)
    rng = np.random.default_rng(4)
    U = rng.normal(size=(C_pos.shape[0], cfg.k))
    V = rng.normal(size=(C_pos.shape[1], cfg.k))

    coo_p, coo_n = C_pos.tocoo(), M_neg.tocoo()
    ottenuta = calcola_perdita(U, V, (coo_p.row, coo_p.col), (coo_n.row, coo_n.col), cfg)
    attesa = _perdita_ingenua(U, V, C_pos, M_neg, cfg)

    assert abs(ottenuta - attesa) < 1e-8 * max(1.0, abs(attesa)), (ottenuta, attesa)


def test_perdita_non_crescente():
    """L'alternanza risolve esattamente il minimo di un blocco per volta.

    Ne segue che la perdita non puo' salire. Qui e' un assert, non una stampa:
    e' il controllo che intercetta un errore nelle equazioni normali.
    """
    cfg = WALSConfig(k=6, n_iter=12, w_neg=0.4)
    C_pos, M_neg = matrici_giocattolo(seed=5)
    _, storia = train(C_pos, M_neg, cfg, verbose=False)

    for prima, dopo in zip(storia, storia[1:]):
        assert dopo <= prima + 1e-9, f"la perdita e' salita: {prima} -> {dopo}"


def test_foldin_riproduce_il_passo_di_training():
    """Il fold-in e' esattamente il ramo utenti ristretto alle celle positive.

    Senza negativi osservati le due cose devono coincidere a precisione
    macchina. E' il requisito del progetto ("accept as input a set of liked
    documents") verificato come identita', non stimato con una metrica.

    Con i negativi presenti i due risultati divergono, ed e' voluto: di un
    utente nuovo non si sa cosa gli sia stato mostrato senza essere cliccato.
    """
    cfg = WALSConfig(k=7, n_iter=6, w_neg=0.3)
    C_pos, M_neg = matrici_giocattolo(seed=6, con_negativi=False)
    modello, _ = train(C_pos, M_neg, cfg, verbose=False)

    pos, neg = C_pos.tocsr(), M_neg.tocsr()
    # U ricalcolata sulla V finale: la U salvata usa la V dell'iterazione prima.
    U_attesa = _risolvi_fattori(modello.V, pos.indptr, pos.indices,
                                neg.indptr, neg.indices, C_pos.shape[0], cfg)

    for i in range(0, C_pos.shape[0], 7):
        profilo = pos.indices[pos.indptr[i]:pos.indptr[i + 1]]
        assert np.allclose(modello.fold_in(profilo), U_attesa[i], atol=1e-10)


def test_determinismo():
    """Stesso seme, stessi numeri: due addestramenti devono coincidere."""
    cfg = WALSConfig(k=5, n_iter=4, seed=42)
    C_pos, M_neg = matrici_giocattolo(seed=7)
    a, storia_a = train(C_pos, M_neg, cfg, verbose=False)
    b, storia_b = train(C_pos, M_neg, cfg, verbose=False)

    assert np.array_equal(a.U, b.U) and np.array_equal(a.V, b.V)
    assert storia_a == storia_b


def test_raccomanda_esclude_il_profilo():
    """Un articolo gia' letto non puo' comparire fra le raccomandazioni."""
    cfg = WALSConfig(k=6, n_iter=5)
    C_pos, M_neg = matrici_giocattolo(seed=8)
    modello, _ = train(C_pos, M_neg, cfg, verbose=False)

    pos = C_pos.tocsr()
    profilo = pos.indices[pos.indptr[0]:pos.indptr[1]]
    top, punteggi = modello.raccomanda(profilo, top_n=10)

    assert not set(top.tolist()) & set(profilo.tolist())
    assert len(top) == 10
    assert np.all(np.diff(punteggi) <= 1e-12), "il ranking non e' ordinato"


def test_metriche_casi_limite():
    """Comportamento delle misure quando tutto e' rilevante o niente lo e'."""
    scores = np.array([5.0, 4.0, 3.0, 2.0])

    tutti = ordina_per_score(np.ones(4), scores)
    assert average_precision(tutti) == 1.0
    assert r_precision(tutti) == 1.0
    assert precision_at_k(tutti, 4) == 1.0
    assert np.allclose(precisione_interpolata(tutti), 1.0)

    nessuno = ordina_per_score(np.zeros(4), scores)
    assert average_precision(nessuno) == 0.0
    assert r_precision(nessuno) == 0.0
    assert np.allclose(precisione_interpolata(nessuno), 0.0)


def test_serializzazione(tmp="artifacts/_test_modello"):
    """Il modello ricaricato da disco deve essere identico a quello salvato."""
    import shutil
    from data_loader import Mappings

    cfg = WALSConfig(k=5, n_iter=3)
    C_pos, M_neg = matrici_giocattolo(seed=9)
    modello, _ = train(C_pos, M_neg, cfg, verbose=False)

    mappings = Mappings(
        user2idx={}, idx2user=np.array([f"U{i}" for i in range(C_pos.shape[0])]),
        news2idx={}, idx2news=np.array([f"N{j}" for j in range(C_pos.shape[1])]))
    try:
        modello.save(tmp, mappings)
        ricaricato, _ = WALSModel.load(tmp)
        assert np.array_equal(modello.U, ricaricato.U)
        assert np.array_equal(modello.V, ricaricato.V)
        assert modello.cfg == ricaricato.cfg
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- Controlli sugli artefatti reali (richiedono il dataset) -------------

def test_artefatti_reali():
    """Le matrici costruite dai dati devono avere i conteggi attesi.

    I valori vengono da un conteggio indipendente fatto con awk sui .tsv, non
    dal codice stesso: se un giorno il parsing cambiasse comportamento, questo
    test se ne accorgerebbe.
    """
    from data_loader import load_artifacts

    art = load_artifacts("artifacts/train")
    C_pos, M_neg = art["C_pos"], art["M_neg"]

    assert C_pos.shape == (50_000, 51_282), C_pos.shape
    assert C_pos.nnz == 1_148_447, C_pos.nnz
    assert M_neg.nnz == 4_746_537, M_neg.nnz
    assert C_pos.multiply(M_neg).nnz == 0, "positivi e negativi si sovrappongono"
    assert len(art["mappings"].idx2news) == 51_282


def test_eval_set_reale():
    """La struttura di valutazione deve coprire tutto il dev senza scarti."""
    from evaluation import load_eval_set

    eval_set = load_eval_set("artifacts/dev/eval_set.npz")
    assert len(eval_set) == 73_152, len(eval_set)
    assert eval_set.stats["candidati"] == 2_740_998
    assert abs(eval_set.stats["quota_cold"] - 0.2003) < 5e-4
    assert eval_set.stats["impression_scartate_senza_click"] == 0


VELOCI = [
    test_metriche,
    test_metriche_casi_limite,
    test_gramiana_ramo_utenti,
    test_gramiana_ramo_item,
    test_perdita_in_forma_chiusa,
    test_perdita_non_crescente,
    test_foldin_riproduce_il_passo_di_training,
    test_determinismo,
    test_raccomanda_esclude_il_profilo,
    test_serializzazione,
]

CON_DATASET = [test_artefatti_reali, test_eval_set_reale]


def esegui(test):
    falliti = 0
    for funzione in test:
        nome = funzione.__name__
        try:
            funzione()
        except AssertionError as errore:
            print(f"  FALLITO  {nome}: {errore}")
            falliti += 1
        except Exception as errore:            # noqa: BLE001 - va segnalato, non nascosto
            print(f"  ERRORE   {nome}: {type(errore).__name__}: {errore}")
            falliti += 1
        else:
            print(f"  ok       {nome}")
    return falliti


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="store_true",
                        help="esegue anche i controlli sugli artefatti reali")
    args = parser.parse_args()

    print("Identita' matematiche e invarianti (matrici giocattolo)")
    falliti = esegui(VELOCI)

    if args.dataset:
        print("\nArtefatti reali")
        falliti += esegui(CON_DATASET)

    totale = len(VELOCI) + (len(CON_DATASET) if args.dataset else 0)
    print(f"\n{totale - falliti}/{totale} test superati")
    return 1 if falliti else 0


if __name__ == "__main__":
    raise SystemExit(main())
