"""Valutazione dei sistemi di raccomandazione sul dev set di MIND.

Protocollo primario: ranking *per impression*. Per ogni impression del dev si
ordinano i candidati effettivamente mostrati all'utente (in media 37) e si
misura quanto in alto finiscono quelli cliccati.

Scelta delle metriche. La letteratura su MIND riporta AUC, MRR e nDCG, ma qui
si adottano le misure viste nel corso: MAP, Precision@k, R-precision e la
curva di precisione interpolata a 11 punti. La sostituzione non e' una
rinuncia, perche' il contesto e' ranking con rilevanza *binaria* (click o non
click), che e' esattamente quello in cui quelle misure sono definite:

- l'MRR e' ridondante: quando un'impression ha un solo click l'Average
  Precision coincide con il reciproco del rango, e sul dev di MIND questo
  accade nel 71,2% dei casi; sul restante 28,8% il MAP e' piu' corretto,
  perche' tiene conto di tutti i click e non solo del primo;
- il vantaggio dell'nDCG e' gestire la rilevanza graduata, che qui non
  esiste.
L'AUC resta implementata ma non compare fra i risultati: e' usata solo come
test di correttezza dell'harness, perche' e' l'unica di queste misure con un
valore atteso noto a priori (uno scorer casuale deve dare 0,5).

Modalita' "full": gli item che non hanno una colonna in V (perche' non
compaiono nel training) non vengono esclusi, ma ricevono uno score di
fallback pari alla popolarita' media. E' la modalita' che rispecchia il
sistema come lo si userebbe davvero, e tiene conto del fatto che sul dev di
MIND circa il 20% dei candidati e' "cold".

Nota strutturale su MIND-small, verificata sui dati: train e dev contengono
50.000 utenti ciascuno ma **non sono gli stessi utenti**, ne condividono solo
5.943. Per l'88% delle impression del dev non esiste quindi una riga in U, e
l'unico modo di produrre un punteggio personalizzato e' ricavare il vettore
utente dal suo insieme di documenti graditi (la History) risolvendo un singolo
passo di WALS a V fissata. Per questo la struttura di valutazione porta con se'
la History di ogni impression: e' l'input previsto dal progetto.

Metrica secondaria: Recall@k su tutto il catalogo, che riflette il caso d'uso
del progetto ("ritorna un ranking di documenti") invece del ri-ordinamento di
una lista gia' preselezionata.
"""

import os
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from data_loader import load_artifacts, load_behaviors

# Seme usato per permutare i candidati di ogni impression: vedi build_eval_set.
SHUFFLE_SEED = 12345

# I livelli di richiamo della 11-point interpolated average precision.
LIVELLI_11P = np.linspace(0.0, 1.0, 11)


# --- Struttura di valutazione -------------------------------------------
#
# Candidati e History sono memorizzati in forma "appiattita" (come una CSR):
# array unici di item e label, piu' array di offset che delimitano le
# impression. E' piu' compatto e piu' veloce da scorrere di liste di array.

@dataclass
class EvalSet:
    train_user_idx: np.ndarray  # (n_impr,) riga in U, -1 se l'utente non e' nel train
    dev_user_idx: np.ndarray    # (n_impr,) indice dell'utente fra quelli del dev
    offsets: np.ndarray         # (n_impr + 1,) delimitatori dei candidati
    items: np.ndarray           # (n_cand,) indice di colonna, -1 se item cold
    labels: np.ndarray          # (n_cand,) 1 = cliccato, 0 = non cliccato
    hist_offsets: np.ndarray    # (n_impr + 1,) delimitatori della History
    hist_items: np.ndarray      # (n_hist,) colonne della History (solo item noti)
    stats: dict                 # conteggi diagnostici sulla costruzione

    def __len__(self) -> int:
        return len(self.dev_user_idx)

    def candidates(self, i: int) -> tuple:
        """Item e label dell'i-esima impression."""
        start, end = self.offsets[i], self.offsets[i + 1]
        return self.items[start:end], self.labels[start:end]

    def history(self, i: int) -> np.ndarray:
        """Profilo dell'utente dell'i-esima impression: l'insieme dei documenti graditi.

        Contiene solo gli item presenti nel training, perche' sono gli unici
        con una colonna in V e quindi gli unici utilizzabili per il fold-in.
        """
        start, end = self.hist_offsets[i], self.hist_offsets[i + 1]
        return self.hist_items[start:end]


def build_eval_set(behaviors_df, mappings) -> EvalSet:
    """Costruisce la struttura di valutazione dalle behaviors del dev.

    Nessuna impression viene scartata per via dell'utente: gli utenti assenti
    dal training ricevono train_user_idx = -1 e vanno valutati tramite la loro
    History. Restano escluse solo le impression prive di click o con tutti i
    candidati cliccati, in cui il ranking non ha nulla da separare; su
    MIND-small non ce ne sono, ma il controllo va tenuto.

    I candidati di ogni impression vengono permutati con un seme fisso. Serve
    a rompere i pareggi in modo non sistematico: tutti gli item cold ricevono
    lo stesso score di fallback, e senza permutazione l'ordine fra loro
    seguirebbe l'ordine in cui MSN li ha mostrati, introducendo un bias.
    """
    rng = np.random.default_rng(SHUFFLE_SEED)

    dev_user2idx = {}
    train_user_idx, dev_user_idx = [], []
    offsets, items, labels = [0], [], []
    hist_offsets, hist_items = [0], []

    n_skipped_no_pos = n_skipped_no_neg = 0
    n_cold = n_candidates = 0
    n_utenti_ignoti = n_history_vuota = 0

    for row in behaviors_df.itertuples(index=False):
        impression_items, impression_labels = [], []
        for token in row.impressions:
            news_id, _, label = token.rpartition("-")
            n_idx = mappings.news2idx.get(news_id)
            impression_items.append(-1 if n_idx is None else n_idx)  # -1 = item cold
            impression_labels.append(1 if label == "1" else 0)

        impression_labels = np.array(impression_labels, dtype=np.int8)
        n_pos = int(impression_labels.sum())
        if n_pos == 0:
            n_skipped_no_pos += 1
            continue
        if n_pos == len(impression_labels):
            n_skipped_no_neg += 1
            continue

        impression_items = np.array(impression_items, dtype=np.int32)
        perm = rng.permutation(len(impression_items))

        u_train = mappings.user2idx.get(row.user_id, -1)
        if u_train < 0:
            n_utenti_ignoti += 1
        u_dev = dev_user2idx.setdefault(row.user_id, len(dev_user2idx))

        profilo = [
            mappings.news2idx[news_id]
            for news_id in row.history
            if news_id in mappings.news2idx
        ]
        if not profilo:
            n_history_vuota += 1

        train_user_idx.append(u_train)
        dev_user_idx.append(u_dev)
        items.append(impression_items[perm])
        labels.append(impression_labels[perm])
        offsets.append(offsets[-1] + len(impression_items))
        hist_items.extend(profilo)
        hist_offsets.append(hist_offsets[-1] + len(profilo))

        n_candidates += len(impression_items)
        n_cold += int((impression_items < 0).sum())

    stats = {
        "impression_valutate": len(dev_user_idx),
        "impression_utente_non_nel_train": n_utenti_ignoti,
        "impression_history_vuota": n_history_vuota,
        "impression_scartate_senza_click": n_skipped_no_pos,
        "impression_scartate_tutte_click": n_skipped_no_neg,
        "utenti_distinti_dev": len(dev_user2idx),
        "candidati": n_candidates,
        "candidati_cold": n_cold,
        "quota_cold": n_cold / n_candidates if n_candidates else 0.0,
    }

    return EvalSet(
        train_user_idx=np.array(train_user_idx, dtype=np.int32),
        dev_user_idx=np.array(dev_user_idx, dtype=np.int32),
        offsets=np.array(offsets, dtype=np.int64),
        items=np.concatenate(items) if items else np.empty(0, dtype=np.int32),
        labels=np.concatenate(labels) if labels else np.empty(0, dtype=np.int8),
        hist_offsets=np.array(hist_offsets, dtype=np.int64),
        hist_items=np.array(hist_items, dtype=np.int32),
        stats=stats,
    )


def save_eval_set(path: str, eval_set: EvalSet) -> None:
    """Serializza la struttura di valutazione per non ricostruirla ogni volta."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        train_user_idx=eval_set.train_user_idx,
        dev_user_idx=eval_set.dev_user_idx,
        offsets=eval_set.offsets,
        items=eval_set.items,
        labels=eval_set.labels,
        hist_offsets=eval_set.hist_offsets,
        hist_items=eval_set.hist_items,
        stats_keys=np.array(list(eval_set.stats.keys())),
        stats_values=np.array(list(eval_set.stats.values()), dtype=np.float64),
    )


def load_eval_set(path: str) -> EvalSet:
    with np.load(path, allow_pickle=False) as data:
        stats = dict(zip(data["stats_keys"].tolist(), data["stats_values"].tolist()))
        return EvalSet(
            train_user_idx=data["train_user_idx"],
            dev_user_idx=data["dev_user_idx"],
            offsets=data["offsets"],
            items=data["items"],
            labels=data["labels"],
            hist_offsets=data["hist_offsets"],
            hist_items=data["hist_items"],
            stats=stats,
        )


# --- Metriche di ranking (§14 del corso) ---------------------------------
#
# Tutte le misure si calcolano dalla stessa quantita': il vettore delle label
# riordinate per score decrescente. Si ordina una volta sola per impression e
# si riusa il risultato, invece di riordinare in ogni metrica.

def ordina_per_score(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Label dei candidati messe in ordine di score decrescente."""
    return labels[np.argsort(-scores, kind="stable")]


def precision_at_k(ordered: np.ndarray, k: int) -> float:
    """P@k: frazione di rilevanti fra i primi k documenti restituiti."""
    k = min(k, len(ordered))
    return float(ordered[:k].sum() / k)


def average_precision(ordered: np.ndarray) -> float:
    """Average Precision: media della precision calcolata a ogni rango rilevante.

    E' la misura che premia i sistemi che collocano i rilevanti in alto:
    spostare un rilevante dal rango 10 al rango 2 alza l'AP anche se
    precision e recall finali restano identiche.
    """
    n_rilevanti = ordered.sum()
    if n_rilevanti == 0:
        return 0.0
    ranghi = np.arange(1, len(ordered) + 1)
    precision = np.cumsum(ordered) / ranghi
    return float(precision[ordered == 1].sum() / n_rilevanti)


def r_precision(ordered: np.ndarray) -> float:
    """R-precision: precisione sui primi R documenti, con R = numero di rilevanti.

    E' il punto di break-even della curva PR, dove precision e recall
    coincidono.
    """
    n_rilevanti = int(ordered.sum())
    if n_rilevanti == 0:
        return 0.0
    return float(ordered[:n_rilevanti].sum() / n_rilevanti)


def precisione_interpolata(ordered: np.ndarray, livelli: np.ndarray = LIVELLI_11P) -> np.ndarray:
    """Precisione interpolata ai livelli di richiamo dati.

    Definizione del corso: P_interp(r) = max{ P(r') : r' >= r }, cioe' la
    precisione piu' alta osservabile a un richiamo pari o superiore a r.
    Serve a eliminare le oscillazioni a dente di sega della curva PR grezza.
    """
    n_rilevanti = ordered.sum()
    if n_rilevanti == 0:
        return np.zeros(len(livelli))

    ranghi = np.arange(1, len(ordered) + 1)
    cumulate = np.cumsum(ordered)
    precision = cumulate / ranghi
    recall = cumulate / n_rilevanti

    # Massimo della precision "da qui in avanti", calcolato in una passata.
    max_successivo = np.maximum.accumulate(precision[::-1])[::-1]

    # recall e' non decrescente: per ogni livello basta il primo rango che lo raggiunge.
    primo_rango = np.searchsorted(recall, livelli, side="left")
    primo_rango = np.minimum(primo_rango, len(ordered) - 1)
    return max_successivo[primo_rango]


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC nella forma equivalente della statistica di Mann-Whitney.

    Non compare fra i risultati: e' usata solo come test di correttezza
    dell'harness, perche' e' l'unica di queste misure il cui valore atteso su
    uno scorer casuale e' noto a priori (0,5). I pareggi vanno gestiti con
    ranghi medi, ed e' la parte piu' facile da sbagliare.
    """
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    ranghi = rankdata(scores, method="average")
    somma_ranghi_pos = float(ranghi[labels == 1].sum())
    return (somma_ranghi_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# --- Modelli di scoring --------------------------------------------------

class Scorer:
    """Interfaccia comune dei modelli da valutare.

    `user_idx` e' la riga in U (-1 se l'utente non compare nel training) e
    `history` e' il suo insieme di documenti graditi: un modello puo' usare
    l'una, l'altra o nessuna delle due. `item_idx` puo' contenere -1 per gli
    item cold, che ogni implementazione sostituisce con il proprio fallback.
    """

    name = "scorer"

    def scores_for(self, user_idx: int, history: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def all_item_scores(self, user_idx: int, history: np.ndarray) -> np.ndarray:
        """Score su tutto il catalogo, per la valutazione full-catalog."""
        raise NotImplementedError


class GlobalScorer(Scorer):
    """Scorer non personalizzato: un unico vettore di score valido per tutti.

    Il fallback per gli item cold e' la media degli score sugli item noti,
    cioe' la "popolarita' media" del catalogo: un item mai visto viene
    trattato come un item di popolarita' tipica, ne' promosso ne' affossato.
    """

    def __init__(self, item_score: np.ndarray, name: str):
        self.item_score = np.asarray(item_score, dtype=np.float64)
        self.fallback = float(self.item_score.mean())
        self.name = name

    def scores_for(self, user_idx, history, item_idx: np.ndarray) -> np.ndarray:
        safe_idx = np.maximum(item_idx, 0)
        return np.where(item_idx >= 0, self.item_score[safe_idx], self.fallback)

    def all_item_scores(self, user_idx, history) -> np.ndarray:
        return self.item_score


class RandomScorer(Scorer):
    """Baseline di controllo: sull'AUC deve dare ~0,500.

    Se non lo fa, il bug e' nelle metriche, non nel modello.
    """

    name = "random"

    def __init__(self, n_items: int, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n_items = n_items

    def scores_for(self, user_idx, history, item_idx: np.ndarray) -> np.ndarray:
        return self.rng.random(len(item_idx))

    def all_item_scores(self, user_idx, history) -> np.ndarray:
        return self.rng.random(self.n_items)


class MFScorer(Scorer):
    """Scorer per la Weighted MF: score(u, i) = u . v_i.

    Dato che l'88% degli utenti del dev non ha una riga in U, il vettore
    utente viene ricavato dal suo profilo tramite `fold_in`, una funzione
    (history -> vettore k) che verra' fornita dal modulo WALS: e' un singolo
    passo dell'alternanza, quello che fissa V e risolve per u. Se il profilo
    e' vuoto e l'utente e' ignoto non c'e' nulla da personalizzare, e si
    ricade sullo scorer di riserva.

    Il fallback per gli item cold e' u . media(V), cioe' lo score che
    l'utente darebbe a un item "medio": l'analogo personalizzato della
    popolarita' media usata dalle baseline.
    """

    def __init__(self, U, V, fold_in=None, fallback_scorer: Scorer = None, name: str = "wmf"):
        self.U = U
        self.V = V
        self.fold_in = fold_in
        self.fallback_scorer = fallback_scorer
        self.mean_item = V.mean(axis=0)
        self.name = name

    def _user_vector(self, user_idx: int, history: np.ndarray):
        if self.fold_in is not None and len(history) > 0:
            return self.fold_in(history)
        if user_idx >= 0:
            return self.U[user_idx]
        return None

    def scores_for(self, user_idx, history, item_idx: np.ndarray) -> np.ndarray:
        u = self._user_vector(user_idx, history)
        if u is None:
            return self.fallback_scorer.scores_for(user_idx, history, item_idx)
        safe_idx = np.maximum(item_idx, 0)
        return np.where(item_idx >= 0, self.V[safe_idx] @ u, float(u @ self.mean_item))

    def all_item_scores(self, user_idx, history) -> np.ndarray:
        u = self._user_vector(user_idx, history)
        if u is None:
            return self.fallback_scorer.all_item_scores(user_idx, history)
        return self.V @ u


class ScorerIbrido(Scorer):
    """Combina due scorer normalizzando i punteggi dentro ogni impression.

    La normalizzazione e' necessaria perche' i due punteggi vivono su scale
    diverse (un CTR sta in [0,1], un prodotto scalare no); farla per
    impression invece che globalmente rende la combinazione indipendente da
    quanto e' "facile" quella particolare lista di candidati.

    Serve a verificare se il segnale collaborativo della fattorizzazione e'
    complementare a quello di popolarita' del CTR, o se e' solo una sua copia
    peggiore.
    """

    def __init__(self, a: Scorer, b: Scorer, peso: float = 1.0, name="ibrido"):
        self.a, self.b, self.peso, self.name = a, b, peso, name

    @staticmethod
    def _z(x):
        s = x.std()
        return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)

    def scores_for(self, user_idx, history, item_idx):
        return (self._z(self.a.scores_for(user_idx, history, item_idx))
                + self.peso * self._z(self.b.scores_for(user_idx, history, item_idx)))

    def all_item_scores(self, user_idx, history):
        return (self._z(self.a.all_item_scores(user_idx, history))
                + self.peso * self._z(self.b.all_item_scores(user_idx, history)))


def popularity_from_feedback(C_pos) -> np.ndarray:
    """Popolarita' = numero di utenti distinti con segnale positivo sull'item.

    Include sia i click nelle impression sia la History, cioe' tutto cio' che
    finisce in C_pos.
    """
    return np.asarray(C_pos.sum(axis=0)).ravel()


def item_impression_stats(behaviors_df, mappings) -> tuple:
    """Conteggi per item delle sole impression: (click, volte mostrato).

    Non si possono ricavare da C_pos e M_neg: quelle contengono coppie
    (utente, item) *distinte*, e i positivi includono la History, che non
    corrisponde a nessuna esposizione osservata. Per il CTR servono i
    conteggi grezzi delle impression.
    """
    n_items = len(mappings.idx2news)
    clicks = np.zeros(n_items, dtype=np.int64)
    shows = np.zeros(n_items, dtype=np.int64)

    for impressions in behaviors_df["impressions"]:
        for token in impressions:
            news_id, _, label = token.rpartition("-")
            idx = mappings.news2idx.get(news_id)
            if idx is None:
                continue
            shows[idx] += 1
            if label == "1":
                clicks[idx] += 1

    return clicks, shows


def smoothed_ctr(clicks: np.ndarray, shows: np.ndarray, prior_weight: float = 10.0) -> np.ndarray:
    """CTR per item con smoothing bayesiano verso il CTR globale.

    Senza smoothing un articolo mostrato 1 volta e cliccato 1 volta avrebbe
    CTR 100% e dominerebbe il ranking. `prior_weight` equivale ad aggiungere
    a ogni item un numero fittizio di esposizioni con il CTR medio globale:
    piu' un item e' raro, piu' il suo punteggio viene tirato verso la media.
    """
    global_ctr = clicks.sum() / shows.sum()
    return (clicks + prior_weight * global_ctr) / (shows + prior_weight)


# --- Ciclo di valutazione ------------------------------------------------

def evaluate(scorer: Scorer, eval_set: EvalSet, ks=(5, 10)) -> tuple:
    """Valuta uno scorer sul dev: restituisce (risultati, curva a 11 punti).

    Le misure sono calcolate su ogni impression e poi mediate, come nel corso
    si mediano sulle query: qui una impression svolge il ruolo di una query.
    """
    n = len(eval_set)
    ap = np.empty(n)
    rprec = np.empty(n)
    patk = {k: np.empty(n) for k in ks}
    curva = np.zeros(len(LIVELLI_11P))

    for i in range(n):
        items, labels = eval_set.candidates(i)
        scores = scorer.scores_for(eval_set.train_user_idx[i], eval_set.history(i), items)
        ordered = ordina_per_score(labels.astype(np.float64), scores)

        ap[i] = average_precision(ordered)
        rprec[i] = r_precision(ordered)
        for k in ks:
            patk[k][i] = precision_at_k(ordered, k)
        curva += precisione_interpolata(ordered)

    curva /= n
    risultati = {"modello": scorer.name, "MAP": ap.mean()}
    for k in ks:
        risultati[f"P@{k}"] = patk[k].mean()
    risultati["R-prec"] = rprec.mean()
    risultati["11-point"] = float(curva.mean())
    return risultati, curva


def evaluate_recall_full_catalog(scorer: Scorer, eval_set: EvalSet, ks=(10, 50, 100),
                                 max_users: int = None) -> dict:
    """Recall@k sull'intero catalogo, non sui soli candidati dell'impression.

    E' la metrica che corrisponde al caso d'uso chiesto dal progetto: dato un
    utente, produrre un ranking di documenti. Gli item gia' presenti nella
    History vengono esclusi dal ranking, altrimenti si raccomanderebbero
    articoli gia' letti; per lo stesso motivo sono esclusi dalla verita' di
    riferimento, che e' l'insieme degli item cliccati dall'utente nel dev,
    aggregando tutte le sue impression.
    """
    profili = {}   # dev_user_idx -> [train_user_idx, history, item cliccati]
    for i in range(len(eval_set)):
        items, labels = eval_set.candidates(i)
        u = int(eval_set.dev_user_idx[i])
        if u not in profili:
            profili[u] = [int(eval_set.train_user_idx[i]), eval_set.history(i), set()]
        profili[u][2].update(items[(labels == 1) & (items >= 0)].tolist())

    utenti = sorted(profili)
    if max_users is not None:
        utenti = utenti[:max_users]

    max_k = max(ks)
    hits = {k: [] for k in ks}

    for u in utenti:
        train_idx, history, cliccati = profili[u]
        rilevanti = cliccati - set(history.tolist())
        if not rilevanti:
            continue

        scores = np.array(scorer.all_item_scores(train_idx, history), dtype=np.float64, copy=True)
        scores[history] = -np.inf  # non raccomandare cio' che ha gia' letto

        top = np.argpartition(-scores, max_k)[:max_k]
        top = top[np.argsort(-scores[top], kind="stable")]
        for k in ks:
            hits[k].append(len(rilevanti & set(top[:k].tolist())) / len(rilevanti))

    risultati = {"modello": scorer.name, "utenti": len(hits[max_k])}
    for k in ks:
        risultati[f"Recall@{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0
    return risultati


# --- Test di correttezza -------------------------------------------------

def test_metriche() -> None:
    """Verifica le metriche su un caso calcolabile a mano.

    Sei candidati ordinati per score decrescente, rilevanti al rango 1 e 3:

        rango      1     2     3     4     5     6
        rilevante  si    no    si    no    no    no
        precision  1.00  0.50  0.67  0.50  0.40  0.33
        recall     0.50  0.50  1.00  1.00  1.00  1.00

    AP        = (1.00 + 0.67) / 2               = 5/6
    P@5       = 2 / 5                           = 0.40
    R-prec    = rilevanti nei primi 2 / 2       = 0.50
    11-point  = (6 * 1.00 + 5 * 0.67) / 11      = 0.8485
    AUC       = coppie concordi / (2 * 4)       = 7/8
    """
    labels = np.array([1, 0, 1, 0, 0, 0], dtype=np.float64)
    scores = np.array([6, 5, 4, 3, 2, 1], dtype=np.float64)
    ordered = ordina_per_score(labels, scores)

    assert abs(average_precision(ordered) - 5 / 6) < 1e-12
    assert abs(precision_at_k(ordered, 5) - 0.4) < 1e-12
    assert abs(r_precision(ordered) - 0.5) < 1e-12
    assert abs(auc_score(labels, scores) - 7 / 8) < 1e-12

    curva = precisione_interpolata(ordered)
    assert np.allclose(curva[:6], 1.0)           # livelli 0.0 - 0.5
    assert np.allclose(curva[6:], 2 / 3)         # livelli 0.6 - 1.0
    assert abs(curva.mean() - (6 + 5 * 2 / 3) / 11) < 1e-12

    # Una lista ordinata al contrario deve dare AP e AUC minimi.
    assert auc_score(labels, -scores) < 0.5
    print("test_metriche: OK")


def test_auc_casuale(eval_set: EvalSet, n_impressioni: int = 5000) -> float:
    """Uno scorer casuale deve dare AUC ~ 0,5: e' il test di correttezza dell'harness."""
    scorer = RandomScorer(n_items=1, seed=0)
    valori = np.empty(min(n_impressioni, len(eval_set)))
    for i in range(len(valori)):
        items, labels = eval_set.candidates(i)
        valori[i] = auc_score(labels.astype(np.float64), scorer.scores_for(-1, None, items))
    return float(valori.mean())


# --- Output --------------------------------------------------------------

def stampa_tabella(righe: list) -> None:
    """Stampa i risultati come tabella allineata."""
    if not righe:
        return
    colonne = list(righe[0].keys())
    larghezze = [max(len(c), max(len(_fmt(r[c])) for r in righe)) for c in colonne]
    print("  ".join(c.ljust(w) for c, w in zip(colonne, larghezze)))
    print("  ".join("-" * w for w in larghezze))
    for r in righe:
        print("  ".join(_fmt(r[c]).ljust(w) for c, w in zip(colonne, larghezze)))


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def grafico_11_punti(curve: dict, path: str) -> None:
    """Salva la curva di precisione interpolata a 11 punti per ogni modello."""
    import matplotlib
    matplotlib.use("Agg")  # nessuna finestra: serve solo il file
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nome, curva in curve.items():
        ax.plot(LIVELLI_11P, curva, marker="o", markersize=4, label=nome)

    ax.set_xlabel("Richiamo")
    ax.set_ylabel("Precisione interpolata")
    ax.set_title("11-point interpolated average precision — dev di MIND-small")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ARTIFACTS = "artifacts/train"
    EVAL_SET_PATH = "artifacts/dev/eval_set.npz"
    STATS_PATH = "artifacts/train/item_stats.npz"
    CURVE_PATH = "artifacts/dev/curva_11_punti.png"

    test_metriche()

    train = load_artifacts(ARTIFACTS)
    mappings, C_pos = train["mappings"], train["C_pos"]

    # Struttura di valutazione: ricostruita solo la prima volta.
    if os.path.exists(EVAL_SET_PATH):
        eval_set = load_eval_set(EVAL_SET_PATH)
        print(f"Struttura di valutazione ricaricata da {EVAL_SET_PATH}")
    else:
        dev_behaviors = load_behaviors("data/dev/behaviors.tsv")
        eval_set = build_eval_set(dev_behaviors, mappings)
        save_eval_set(EVAL_SET_PATH, eval_set)
        print(f"Struttura di valutazione costruita e salvata in {EVAL_SET_PATH}")

    print("\nCostruzione del dev set:")
    for chiave, valore in eval_set.stats.items():
        riga = f"{valore:,.4f}" if chiave == "quota_cold" else f"{valore:,.0f}"
        print(f"  {chiave:34s} {riga:>12s}")

    print(f"\ntest_auc_casuale (atteso ~0.5): {test_auc_casuale(eval_set):.4f}")

    # Conteggi per il CTR: richiedono una passata sulle behaviors di training.
    if os.path.exists(STATS_PATH):
        with np.load(STATS_PATH) as d:
            clicks, shows = d["clicks"], d["shows"]
    else:
        train_behaviors = load_behaviors("data/train/behaviors.tsv")
        clicks, shows = item_impression_stats(train_behaviors, mappings)
        np.savez(STATS_PATH, clicks=clicks, shows=shows)

    baselines = [
        RandomScorer(n_items=len(mappings.idx2news), seed=0),
        GlobalScorer(popularity_from_feedback(C_pos), name="popolarita"),
        GlobalScorer(clicks.astype(np.float64), name="click impression"),
        GlobalScorer(smoothed_ctr(clicks, shows), name="CTR smussato"),
    ]

    righe, curve = [], {}
    for scorer in baselines:
        risultati, curva = evaluate(scorer, eval_set)
        righe.append(risultati)
        curve[scorer.name] = curva

    print("\nProtocollo per impression (modalita' full, fallback = popolarita' media)")
    stampa_tabella(righe)

    grafico_11_punti(curve, CURVE_PATH)
    np.savez("artifacts/dev/curve_11_punti.npz", livelli=LIVELLI_11P, **curve)
    print(f"\nCurva a 11 punti salvata in {CURVE_PATH}")

    print("\nProtocollo full-catalog (campione di 2000 utenti)")
    stampa_tabella([evaluate_recall_full_catalog(s, eval_set, max_users=2000) for s in baselines])
