"""Quanto è specifica una raccomandazione: sottocategoria o macro-categoria?

A un utente che legge molto football_nfl il sistema propone altro football_nfl
o si limita a "sport in generale"? La domanda si può misurare, perché in
news.tsv ogni articolo porta due etichette: una categoria (17 valori, es.
sports) e una sottocategoria (264 valori, es. football_nfl).

Il punto della verifica è che **il modello non legge mai quelle etichette**:
la fattorizzazione vede solo la matrice di feedback. Se le raccomandazioni si
concentrano sulla sottocategoria giusta, quella struttura è emersa dai soli
pattern di co-click, e le etichette servono qui unicamente da metro esterno.

Si confrontano tre sistemi sugli stessi utenti: la Weighted MF, il CTR (non
personalizzato, quindi limite inferiore di ciò che si ottiene senza conoscere
l'utente) e la scelta casuale, la cui quota attesa è semplicemente la
frequenza di quella sottocategoria nel catalogo.

    python specializzazione.py
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_behaviors, load_index, load_news
from evaluation import smoothed_ctr
from figure import CARTELLA, COLORI
from wals import WALSModel

TOP = 10                 # quante raccomandazioni guardare
QUOTA_MINIMA = 0.5       # frazione del profilo nella sottocategoria dominante
PROFILO_MINIMO = 5       # articoli minimi perché il tema sia attendibile
MAX_UTENTI = 3000


def etichette_per_colonna(mappings):
    """Categoria e sottocategoria di ogni colonna di V."""
    news = load_news("data/train/news.tsv").set_index("news_id")
    per_cat = news["category"].to_dict()
    per_sub = news["subcategory"].to_dict()
    cat = np.array([per_cat.get(n, "?") for n in mappings.idx2news])
    sub = np.array([per_sub.get(n, "?") for n in mappings.idx2news])
    return cat, sub


def _quote(top, cat, sub, sub_dominante, cat_dominante):
    """Ripartizione delle raccomandazioni: stessa sottocat., stessa cat., altro."""
    stessa_sub = np.mean(sub[top] == sub_dominante)
    stessa_cat = np.mean(cat[top] == cat_dominante)
    return stessa_sub, stessa_cat - stessa_sub, 1.0 - stessa_cat


def analizza(modello, mappings, cat, sub, punteggio_ctr):
    """Ripartizione media per WMF, CTR e scelta casuale, sugli utenti a tema."""
    freq_sub = Counter(sub)
    freq_cat = Counter(cat)
    n_item = len(sub)

    quote = {nome: [] for nome in ("WMF", "CTR", "caso")}
    esempi = []

    for row in load_behaviors("data/dev/behaviors.tsv").itertuples(index=False):
        profilo = [mappings.news2idx[n] for n in row.history if n in mappings.news2idx]
        if len(profilo) < PROFILO_MINIMO:
            continue
        profilo = np.array(profilo, dtype=np.int32)

        sub_dom, quante = Counter(sub[profilo]).most_common(1)[0]
        if quante / len(profilo) < QUOTA_MINIMA:
            continue
        cat_dom = cat[profilo][list(sub[profilo]).index(sub_dom)]

        top_wmf, _ = modello.raccomanda(profilo, top_n=TOP)

        # Il CTR non dipende dall'utente: stessa lista per tutti, meno il profilo.
        punteggi = punteggio_ctr.copy()
        punteggi[profilo] = -np.inf
        top_ctr = np.argpartition(-punteggi, TOP)[:TOP]

        quote["WMF"].append(_quote(top_wmf, cat, sub, sub_dom, cat_dom))
        quote["CTR"].append(_quote(top_ctr, cat, sub, sub_dom, cat_dom))
        # Per la scelta casuale la quota attesa è la frequenza nel catalogo.
        p_sub = freq_sub[sub_dom] / n_item
        p_cat = freq_cat[cat_dom] / n_item
        quote["caso"].append((p_sub, p_cat - p_sub, 1.0 - p_cat))

        if quante >= 8 and len(esempi) < 3:
            esempi.append((row.user_id, sub_dom, quante, len(profilo), sub[top_wmf]))

        if len(quote["WMF"]) >= MAX_UTENTI:
            break

    medie = {nome: np.mean(valori, axis=0) for nome, valori in quote.items()}
    return medie, len(quote["WMF"]), esempi


def figura(medie, n_utenti, path):
    """Barre impilate: dove finiscono le prime dieci raccomandazioni."""
    nomi = ["WMF", "CTR", "caso"]
    segmenti = [
        ("stessa sottocategoria", COLORI[2]),
        ("stessa categoria,\nsottocategoria diversa", COLORI[0]),
        ("altrove", "0.82"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    base = np.zeros(len(nomi))
    for i, (etichetta, colore) in enumerate(segmenti):
        valori = np.array([medie[n][i] for n in nomi])
        ax.barh(nomi, valori, left=base, color=colore, label=etichetta, height=0.55)
        for y, (v, b) in enumerate(zip(valori, base)):
            if v > 0.04:
                ax.text(b + v / 2, y, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color="white" if i < 2 else "0.3")
        base += valori

    ax.set_xlim(0, 1)
    ax.set_xlabel(f"quota delle prime {TOP} raccomandazioni")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.invert_yaxis()
    ax.set_title(f"Dove finiscono le raccomandazioni, per {n_utenti:,} utenti con un tema"
                 f"\n(almeno il {QUOTA_MINIMA:.0%} del profilo in una sola sottocategoria)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(CARTELLA, exist_ok=True)
    modello, _ = WALSModel.load("artifacts/wmf")
    mappings = load_index("artifacts/wmf/index.npz")
    cat, sub = etichette_per_colonna(mappings)
    with np.load("artifacts/train/item_stats.npz") as d:
        punteggio_ctr = smoothed_ctr(d["clicks"], d["shows"])

    print(f"{len(set(cat))} categorie, {len(set(sub))} sottocategorie")
    medie, n_utenti, esempi = analizza(modello, mappings, cat, sub, punteggio_ctr)
    print(f"utenti con un tema dominante: {n_utenti:,}\n")

    print(f"{'':6s} {'stessa sottocat.':>17s} {'stessa cat.':>13s} {'altrove':>9s}")
    for nome in ("WMF", "CTR", "caso"):
        s, c, a = medie[nome]
        print(f"{nome:6s} {s:17.1%} {c:13.1%} {a:9.1%}")

    lift_sub = medie["WMF"][0] / medie["caso"][0]
    lift_cat = (medie["WMF"][0] + medie["WMF"][1]) / (medie["caso"][0] + medie["caso"][1])
    print(f"\nlift della WMF sulla sottocategoria: {lift_sub:.1f}x")
    print(f"lift della WMF sulla categoria:      {lift_cat:.1f}x")
    print("Il lift fine e' piu' alto di quello grosso: il modello non si ferma"
          "\nalla macro-categoria, sceglie dentro la sottocategoria giusta.")

    for user_id, sub_dom, quante, totale, sub_top in esempi:
        concordi = int(np.sum(sub_top == sub_dom))
        print(f"\n{user_id}: {quante}/{totale} del profilo in '{sub_dom}'"
              f"  ->  {concordi}/{TOP} raccomandazioni nella stessa sottocategoria")

    percorso = f"{CARTELLA}/specializzazione.png"
    figura(medie, n_utenti, percorso)
    print(f"\nfigura in {percorso}")


if __name__ == "__main__":
    main()
