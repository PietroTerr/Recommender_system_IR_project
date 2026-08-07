"""Genera le figure per la presentazione in artifacts/figure/.

    python figure.py            # le quattro figure che non richiedono training
    python figure.py --sweep    # aggiunge la curva su w_neg (~10 minuti)

I dati calcolati vengono messi in cache in artifacts/figure/dati.npz, cosi'
ritoccare l'aspetto di un grafico non costa un altro giro di valutazione.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # nessuna finestra: servono solo i file
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_artifacts, load_news
from evaluation import (GlobalScorer, RandomScorer, ScorerIbrido, evaluate,
                        load_eval_set, popularity_from_feedback, smoothed_ctr)
from wals import WALSConfig, WALSModel
from wals import train as train_wals

CARTELLA = "artifacts/figure"
CACHE = os.path.join(CARTELLA, "dati.npz")

# Palette stabile fra le figure: categorie sempre dello stesso colore.
COLORI = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C"]


def _categorie_per_colonna(mappings):
    """Categoria di ciascuna colonna di V, allineata agli indici del modello."""
    news = load_news("data/train/news.tsv").set_index("news_id")
    per_id = news["category"].to_dict()
    return np.array([per_id.get(n, "?") for n in mappings.idx2news])


# --- 1a. Similarita' fra categorie nello spazio latente ------------------

def figura_categorie(V, categorie, path, n_categorie=12):
    """Coseno fra i centroidi delle categorie nello spazio dei fattori latenti.

    Se la fattorizzazione ha imparato struttura tematica, il blocco degli
    argomenti affini deve staccarsi dal resto. E' la verifica piu' diretta
    che gli embedding non siano rumore, e non dipende da nessuna metrica di
    ranking.
    """
    nomi, conteggi = np.unique(categorie, return_counts=True)
    nomi = nomi[np.argsort(-conteggi)][:n_categorie]

    centroidi = np.array([V[categorie == c].mean(axis=0) for c in nomi])
    norme = np.linalg.norm(centroidi, axis=1, keepdims=True)
    norme[norme == 0] = 1.0
    S = (centroidi / norme) @ (centroidi / norme).T

    # La diagonale vale 1 per costruzione e non porta informazione: lasciarla
    # dentro schiaccerebbe la scala di colore su cui si legge tutto il resto,
    # che vive in un intervallo molto piu' stretto e tutto positivo.
    diagonale = np.eye(len(S), dtype=bool)
    vmin, vmax = S[~diagonale].min(), S[~diagonale].max()

    palette = plt.get_cmap("YlOrRd").copy()
    palette.set_bad("0.92")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(np.where(diagonale, np.nan, S), cmap=palette, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(nomi)), nomi, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(nomi)), nomi, fontsize=9)
    soglia = vmin + 0.65 * (vmax - vmin)
    for i in range(len(nomi)):
        for j in range(len(nomi)):
            if i != j:
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if S[i, j] > soglia else "black")

    ax.set_title("Similarità coseno fra categorie\nnello spazio dei fattori latenti")
    fig.colorbar(im, ax=ax, shrink=0.75, label="coseno fra i centroidi")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- 1b. Proiezione 2D dei fattori ---------------------------------------

def figura_pca(V, categorie, path, n_categorie=6, campione=8000, seed=0):
    """Prime due componenti principali di V, colorate per categoria.

    PCA calcolata sulla matrice di covarianza 32x32 invece che con una SVD
    su 51.282 righe: e' lo stesso risultato a costo trascurabile.

    I vettori vengono prima normalizzati a lunghezza uno. Senza questo passo
    la prima componente cattura in gran parte la *norma* di v_j, che cresce
    con la popolarita' dell'articolo: si ottiene un ventaglio in cui tutte le
    categorie si sovrappongono. Normalizzando si proiettano le direzioni,
    cioe' la stessa geometria su cui e' costruita la figura dei centroidi.

    Attenzione all'interpretazione: comprimere 32 dimensioni in 2 conserva
    solo la quota di varianza indicata sugli assi. Una sovrapposizione fra
    categorie qui non dimostra che nello spazio pieno siano confuse.
    """
    norme = np.linalg.norm(V, axis=1, keepdims=True)
    norme[norme == 0] = 1.0
    V = V / norme

    Vc = V - V.mean(axis=0)
    autovalori, autovettori = np.linalg.eigh(Vc.T @ Vc / len(Vc))
    ordine = np.argsort(-autovalori)
    componenti = autovettori[:, ordine[:2]]
    quota = autovalori[ordine[:2]] / autovalori.sum()
    proiezione = Vc @ componenti

    nomi, conteggi = np.unique(categorie, return_counts=True)
    principali = nomi[np.argsort(-conteggi)][:n_categorie]

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for colore, categoria in zip(COLORI, principali):
        idx = np.flatnonzero(categorie == categoria)
        if len(idx) > campione // n_categorie:
            idx = rng.choice(idx, campione // n_categorie, replace=False)
        ax.scatter(proiezione[idx, 0], proiezione[idx, 1], s=4, alpha=0.35,
                   color=colore, label=f"{categoria} ({len(idx)})", linewidths=0)

    ax.set_xlabel(f"prima componente ({quota[0]:.1%} della varianza)")
    ax.set_ylabel(f"seconda componente ({quota[1]:.1%})")
    ax.set_title("Proiezione dei fattori degli articoli")
    legenda = ax.legend(markerscale=3, fontsize=9, loc="best")
    for handle in legenda.legend_handles:
        handle.set_alpha(1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- 2. Coda lunga -------------------------------------------------------

def figura_coda_lunga(C_pos, path):
    """Click per articolo e profilo per utente, ordinati e in scala log-log.

    Serve a rendere visibile perche' il problema e' difficile: una densita'
    dello 0,045% e una distribuzione in cui pochi articoli assorbono gran
    parte dei click.
    """
    per_item = np.sort(np.asarray(C_pos.sum(axis=0)).ravel())[::-1]
    per_utente = np.sort(np.asarray(C_pos.sum(axis=1)).ravel())[::-1]

    fig, assi = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dati, titolo, etichetta, colore in [
        (assi[0], per_item, "Articoli", "utenti che l'hanno cliccato", COLORI[0]),
        (assi[1], per_utente, "Utenti", "articoli nel profilo", COLORI[1]),
    ]:
        positivi = dati[dati > 0]
        ax.loglog(np.arange(1, len(positivi) + 1), positivi, color=colore, lw=1.6)
        ax.set_xlabel(f"rango ({len(positivi):,} con almeno un click)")
        ax.set_ylabel(etichetta)
        ax.set_title(titolo)
        ax.grid(alpha=0.3, which="both")

    quota = per_item[:int(0.01 * len(per_item))].sum() / per_item.sum()
    assi[0].annotate(f"l'1% degli articoli\nraccoglie il {quota:.0%} dei positivi",
                     xy=(0.35, 0.82), xycoords="axes fraction", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))

    fig.suptitle("Distribuzione del feedback positivo (1.148.447 coppie, densità 0,045%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- 3. MAP dei modelli, con il pavimento del caso -----------------------

def figura_map(nomi, valori, path):
    """Barre del MAP con la linea del caso.

    La linea orizzontale e' il punto: con ~37 candidati e 1,5 click medi uno
    scorer casuale ottiene gia' MAP 0,231. Senza quel riferimento un MAP di
    0,29 e' illeggibile.
    """
    pavimento = valori[list(nomi).index("random")]
    colori = [COLORI[7] if n == "random" else
              (COLORI[2] if v >= max(valori) - 1e-9 else COLORI[0])
              for n, v in zip(nomi, valori)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    barre = ax.bar(nomi, valori, color=colori, width=0.6)
    ax.axhline(pavimento, color=COLORI[3], ls="--", lw=1.4,
               label=f"scorer casuale ({pavimento:.4f})")

    for barra, v in zip(barre, valori):
        guadagno = (v / pavimento - 1) * 100
        ax.text(barra.get_x() + barra.get_width() / 2, v + 0.004,
                f"{v:.4f}\n{guadagno:+.0f}%", ha="center", fontsize=8.5)

    ax.set_ylabel("MAP")
    ax.set_ylim(0, max(valori) * 1.22)
    ax.set_title("MAP sul dev set, protocollo per impression")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- 4. Effetto del peso dei negativi osservati --------------------------

def sweep_w_neg(C_pos, M_neg, eval_set, valori_w_neg, n_iter=10):
    """Addestra un modello per ogni valore di w_neg e ne misura MAP e R-prec."""
    from confronto import ScorerWMF

    mappe, rprec = [], []
    for w_neg in valori_w_neg:
        cfg = WALSConfig(w_neg=float(w_neg), n_iter=n_iter)
        modello, _ = train_wals(C_pos, M_neg, cfg, verbose=False)
        risultati, _ = evaluate(ScorerWMF(modello), eval_set)
        mappe.append(risultati["MAP"])
        rprec.append(risultati["R-prec"])
        print(f"  w_neg={w_neg:<6.3f}  MAP {risultati['MAP']:.4f}  "
              f"R-prec {risultati['R-prec']:.4f}", flush=True)
    return np.array(mappe), np.array(rprec)


def figura_w_neg(valori, mappe, rprec, w0, path):
    """MAP e R-precision al variare del peso dei negativi osservati.

    A sinistra della linea verticale il modello non distingue "mostrato e
    ignorato" da "mai mostrato": e' la Weighted MF classica a due livelli.
    Tutto cio' che sta a destra e' guadagno ottenuto dagli impression log.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogx(valori, mappe, marker="o", color=COLORI[0], label="MAP")
    ax.set_xlabel("peso dei negativi osservati  $w_{neg}$  (scala logaritmica)")
    ax.set_ylabel("MAP", color=COLORI[0])
    ax.tick_params(axis="y", labelcolor=COLORI[0])

    destra = ax.twinx()
    destra.semilogx(valori, rprec, marker="s", color=COLORI[1], label="R-precision")
    destra.set_ylabel("R-precision", color=COLORI[1])
    destra.tick_params(axis="y", labelcolor=COLORI[1])

    ax.axvline(w0, color="0.4", ls="--", lw=1.3)
    # Le curve salgono da sinistra a destra: l'angolo in alto a sinistra e'
    # l'unico spazio libero in cui l'annotazione non copre nulla.
    ax.annotate("a sinistra della linea gli impression log\n"
                "non portano informazione: $w_{neg} = w_0$\n"
                "equivale alla Weighted MF a due livelli",
                xy=(0.03, 0.88), xycoords="axes fraction", fontsize=8.5, color="0.25",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.9))

    ax.set_title("Effetto del feedback a tre livelli")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true",
                        help="riesegue la ricerca su w_neg (~10 minuti)")
    args = parser.parse_args()

    os.makedirs(CARTELLA, exist_ok=True)
    art = load_artifacts("artifacts/train")
    mappings, C_pos, M_neg = art["mappings"], art["C_pos"], art["M_neg"]
    modello, _ = WALSModel.load("artifacts/wmf")
    eval_set = load_eval_set("artifacts/dev/eval_set.npz")
    categorie = _categorie_per_colonna(mappings)

    cache = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}

    print("1. struttura dello spazio latente")
    figura_categorie(modello.V, categorie, f"{CARTELLA}/categorie.png")
    figura_pca(modello.V, categorie, f"{CARTELLA}/pca_fattori.png")

    print("2. coda lunga del feedback")
    figura_coda_lunga(C_pos, f"{CARTELLA}/coda_lunga.png")

    print("3. MAP dei modelli")
    if "map_nomi" not in cache:
        from confronto import ScorerWMF
        with np.load("artifacts/train/item_stats.npz") as d:
            ctr = GlobalScorer(smoothed_ctr(d["clicks"], d["shows"]), name="CTR smussato")
        wmf = ScorerWMF(modello)
        scorers = [
            RandomScorer(n_items=len(mappings.idx2news), seed=0),
            GlobalScorer(popularity_from_feedback(C_pos), name="popolarità"),
            ctr,
            wmf,
            ScorerIbrido(wmf, ctr, name="ibrido"),
        ]
        nomi, valori = [], []
        for s in scorers:
            risultati, _ = evaluate(s, eval_set)
            nomi.append(s.name)
            valori.append(risultati["MAP"])
            print(f"  {s.name:16s} {risultati['MAP']:.4f}")
        cache["map_nomi"] = np.array(nomi)
        cache["map_valori"] = np.array(valori)
        np.savez(CACHE, **cache)
    figura_map(cache["map_nomi"], cache["map_valori"], f"{CARTELLA}/map_modelli.png")

    print("4. effetto di w_neg")
    if args.sweep or "w_neg" not in cache:
        if not args.sweep:
            print("   (nessuna cache: eseguo la ricerca, richiede ~10 minuti)")
        valori_w_neg = np.array([0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
        mappe, rprec = sweep_w_neg(C_pos, M_neg, eval_set, valori_w_neg)
        cache.update(w_neg=valori_w_neg, w_neg_map=mappe, w_neg_rprec=rprec)
        np.savez(CACHE, **cache)
    figura_w_neg(cache["w_neg"], cache["w_neg_map"], cache["w_neg_rprec"],
                 WALSConfig().w0, f"{CARTELLA}/effetto_w_neg.png")

    print(f"\nfigure in {CARTELLA}/")


if __name__ == "__main__":
    main()
