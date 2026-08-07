"""Interfaccia web locale per la demo del sistema di raccomandazione.

    python demo_web.py                  # apre il browser su http://localhost:8000
    python demo_web.py --porta 8080 --no-browser

Serve a mostrare dal vivo il requisito del progetto: si compone un utente
scegliendo gli articoli che gli piacciono e il sistema restituisce un ranking
sull'intero catalogo. Il profilo si costruisce cercando gli articoli per
titolo, oppure caricando quello di un utente reale del dataset.

Usa solo la libreria standard (http.server): nessuna dipendenza aggiuntiva
oltre a quelle gia' necessarie al modello.
"""

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from data_loader import load_index, load_news
from wals import WALSModel

RADICE = Path(__file__).resolve().parent
MAX_RISULTATI = 25

# Sotto questa quota il profilo non sposta il vettore utente in modo
# apprezzabile. Il valore e' tarato su misure fatte con profili "LeBron" di
# forza crescente, guardando quanti dei primi dieci risultati restano in tema:
#
#     peso 0,10%  (3 articoli letti da una persona sola)   ->  rumore
#     peso 1,35%  (1 articolo molto letto)                 ->  5/10 in tema
#     peso 3,28%  (3 articoli molto letti)                 ->  9/10 in tema
#     peso 4,39%  (6 articoli molto letti)                 ->  9/10, satura
#
# Il salto di qualita' sta fra 1,35% e 3,28%: la soglia cade in mezzo.
SOGLIA_PROFILO_DEBOLE = 0.02


class Catalogo:
    """Modello, indice e metadati degli articoli, caricati una volta sola."""

    def __init__(self, cartella_modello: str):
        cartella = RADICE / cartella_modello
        if not (cartella / "V.npy").exists():
            sys.exit(f"nessun modello in {cartella}: eseguire prima wals.py")

        self.modello, _ = WALSModel.load(str(cartella))
        self.mappings = load_index(str(cartella / "index.npz"))

        news = load_news(str(RADICE / "data/train/news.tsv")).set_index("news_id")
        dev = load_news(str(RADICE / "data/dev/news.tsv")).set_index("news_id")
        self.titolo = {**dev["title"].to_dict(), **news["title"].to_dict()}
        self.categoria = {**dev["category"].to_dict(), **news["category"].to_dict()}
        self.sottocategoria = {**dev["subcategory"].to_dict(), **news["subcategory"].to_dict()}

        # Titoli in minuscolo, allineati agli indici del modello: la ricerca e'
        # una scansione lineare su 51.282 stringhe, qualche millisecondo.
        self.id_per_indice = self.mappings.idx2news
        self.titoli_ricerca = [self.titolo.get(n, "").lower() for n in self.id_per_indice]
        self.indice_per_id = {n: i for i, n in enumerate(self.id_per_indice)}

        # La norma di v_j misura quanto un articolo pesa nel fold-in: un
        # articolo con pochi click viene schiacciato dalla regolarizzazione e
        # aggiungerlo al profilo non sposta il vettore utente. Serve sia a
        # ordinare la ricerca sia a mostrare la forza del segnale.
        self.norme = np.linalg.norm(self.modello.V, axis=1)

        percorso_feedback = RADICE / "artifacts/train/C_pos.npz"
        if percorso_feedback.exists():
            from scipy.sparse import load_npz
            self.click = np.asarray(load_npz(percorso_feedback).sum(axis=0)).ravel()
        else:
            self.click = None

        # Costanti del sistema del fold-in: con esse la pagina calcola da sola
        # quanto pesa il profilo, senza interrogare il server a ogni modifica.
        cfg = self.modello.cfg
        V = self.modello.V
        self.traccia_comune = float(np.trace(cfg.w0 * (V.T @ V) + cfg.reg * np.eye(cfg.k)))
        self.fattore_correzione = float(cfg.w_pos - cfg.w0)

    def scheda(self, news_id: str, punteggio: float = None) -> dict:
        indice = self.indice_per_id.get(news_id)
        voce = {
            "id": news_id,
            "titolo": self.titolo.get(news_id, "(titolo non disponibile)"),
            "categoria": self.categoria.get(news_id, "?"),
            "sottocategoria": self.sottocategoria.get(news_id, "?"),
            "norma": round(float(self.norme[indice]), 3) if indice is not None else 0.0,
            "click": int(self.click[indice]) if (self.click is not None and indice is not None) else None,
        }
        if punteggio is not None:
            voce["punteggio"] = round(float(punteggio), 3)
        return voce

    def cerca(self, query: str) -> list:
        """Titoli che contengono la query, ordinati per forza del segnale.

        Filtro e ordinamento sono due passaggi distinti: l'ordinamento agisce
        solo dentro l'insieme gia' filtrato, quindi non puo' far entrare
        articoli che non corrispondono alla ricerca. Serve a evitare che in
        cima finiscano articoli letti da una persona sola, il cui vettore
        latente e' quasi nullo e che quindi non contribuiscono al profilo.
        """
        query = query.strip().lower()
        if len(query) < 2:
            return []
        corrispondenze = [i for i, titolo in enumerate(self.titoli_ricerca) if query in titolo]
        corrispondenze.sort(key=lambda i: -self.norme[i])
        return [self.scheda(str(self.id_per_indice[i])) for i in corrispondenze[:MAX_RISULTATI]]

    def profilo_utente(self, user_id: str) -> dict:
        """Cerca la History dell'utente nei due split, con uscita anticipata."""
        for split in ("dev", "train"):
            percorso = RADICE / f"data/{split}/behaviors.tsv"
            if not percorso.exists():
                continue
            with open(percorso, encoding="utf-8") as f:
                for riga in f:
                    campi = riga.rstrip("\n").split("\t")
                    if len(campi) >= 4 and campi[1] == user_id:
                        return {"trovato": True, "split": split,
                                "nel_training": user_id in self.mappings.user2idx,
                                "news": campi[3].split()}
        return {"trovato": False}

    def raccomanda(self, news_ids: list, quanti: int) -> dict:
        """Fold-in dal profilo e ranking sull'intero catalogo."""
        noti, ignoti = [], []
        for n in news_ids:
            (noti if n in self.mappings.news2idx else ignoti).append(n)
        if not noti:
            return {"errore": "Nessuno degli articoli scelti compare nel training: "
                              "senza almeno un articolo noto non e' possibile "
                              "collocare l'utente nello spazio latente."}

        indici = np.array([self.mappings.news2idx[n] for n in noti], dtype=np.int32)
        inizio = time.perf_counter()
        posizioni, punteggi = self.modello.raccomanda(indici, top_n=quanti)
        millisecondi = (time.perf_counter() - inizio) * 1000

        return {
            "profilo": [self.scheda(n) for n in noti],
            "ignorati": [self.scheda(n) for n in ignoti],
            "ranking": [self.scheda(str(self.id_per_indice[i]), s)
                        for i, s in zip(posizioni, punteggi)],
            "millisecondi": round(millisecondi, 2),
            "catalogo": len(self.id_per_indice),
        }


PAGINA = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recommender MIND — demo</title>
<style>
  :root { --bordo:#d8d8d8; --tenue:#666; --acc:#2f6db5; --sfondo:#f6f7f9; }
  * { box-sizing: border-box; }
  body { margin:0; font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
         color:#1b1b1b; background:var(--sfondo); }
  header { background:#fff; border-bottom:1px solid var(--bordo); padding:14px 22px; }
  header h1 { margin:0; font-size:19px; font-weight:600; }
  header p { margin:3px 0 0; color:var(--tenue); font-size:13.5px; }
  main { display:grid; grid-template-columns: 1fr 1fr; gap:20px; padding:20px; }
  @media (max-width: 950px) { main { grid-template-columns: 1fr; } }
  section { background:#fff; border:1px solid var(--bordo); border-radius:8px; padding:16px; }
  h2 { margin:0 0 12px; font-size:15px; text-transform:uppercase;
       letter-spacing:.05em; color:var(--tenue); font-weight:600; }
  input { width:100%; padding:9px 11px; font-size:15px; border:1px solid var(--bordo);
          border-radius:6px; font-family:inherit; }
  .riga { display:flex; gap:8px; align-items:center; }
  .riga input { flex:1; }
  button { padding:9px 14px; font-size:14px; border:1px solid var(--bordo);
           background:#fff; border-radius:6px; cursor:pointer; font-family:inherit; }
  button:hover { background:#eef2f7; border-color:var(--acc); }
  button.primario { background:var(--acc); color:#fff; border-color:var(--acc); }
  button.primario:hover { background:#25599a; }
  .voce { display:flex; gap:10px; align-items:flex-start; padding:8px 0;
          border-bottom:1px solid #eee; }
  .voce:last-child { border-bottom:none; }
  .voce .testo { flex:1; min-width:0; }
  .titolo { font-size:14.5px; }
  .etichette { font-size:12px; color:var(--tenue); margin-top:2px; }
  .cat { background:#eef2f7; border-radius:3px; padding:1px 6px; margin-right:5px; }
  .rango { font-variant-numeric:tabular-nums; font-weight:600; color:var(--acc);
           min-width:2.2em; text-align:right; }
  .punti { font-variant-numeric:tabular-nums; color:var(--tenue); font-size:13px; }
  .vuoto { color:var(--tenue); font-size:14px; padding:10px 0; }
  .nota { font-size:13px; color:var(--tenue); margin-top:12px; }
  .errore { color:#b3261e; font-size:14px; }
  .badge { display:inline-block; font-size:12px; padding:2px 8px; border-radius:10px;
           background:#e7f0fb; color:var(--acc); }
  .badge.attenzione { background:#fdf0e3; color:#8a5300; }
  .scorri { max-height:340px; overflow-y:auto; }
  .avviso { display:none; background:#fdf6e7; border:1px solid #f0dcae; color:#7a5a12;
            border-radius:6px; padding:10px 12px; font-size:13px; margin-top:12px; }
  .avviso b { color:#5f450e; }
  .forza { font-size:11px; letter-spacing:1px; margin-right:6px; }
  .forza .pieno { color:#2f8f4e; }
  .forza .vuoto { color:#c9c9c9; }
</style>
</head>
<body>
<header>
  <h1>Sistema di raccomandazione — Weighted Matrix Factorisation su MIND</h1>
  <p>Si compone un utente come insieme di documenti graditi; il sistema restituisce
     un ranking sull'intero catalogo.</p>
</header>

<main>
  <section>
    <h2>1 · Costruisci il profilo</h2>
    <div class="riga">
      <input id="ricerca" placeholder="cerca un articolo per titolo (es. NFL, Trump, recipe)" autofocus>
    </div>
    <div id="risultati" class="scorri"></div>

    <div class="riga" style="margin-top:14px">
      <input id="utente" placeholder="oppure carica un utente del dataset (es. U19351)">
      <button onclick="caricaUtente()">Carica</button>
    </div>
    <div id="statoUtente"></div>
  </section>

  <section>
    <h2>2 · Documenti graditi <span id="conta" class="badge">0</span></h2>
    <div id="profilo" class="scorri"><div class="vuoto">Nessun articolo scelto.</div></div>
    <div id="avviso" class="avviso"></div>
    <div class="riga" style="margin-top:14px">
      <button class="primario" onclick="raccomanda()">Raccomanda</button>
      <button onclick="svuota()">Svuota</button>
      <span class="nota" style="margin:0">top
        <input id="quanti" type="number" value="10" min="1" max="50"
               style="width:64px;display:inline-block;padding:5px"></span>
    </div>
  </section>

  <section style="grid-column:1 / -1">
    <h2>3 · Ranking dei documenti</h2>
    <div id="ranking"><div class="vuoto">In attesa di un profilo.</div></div>
    <div id="tempi" class="nota"></div>
  </section>
</main>

<script>
let profilo = [];   // schede degli articoli scelti
const K = __COSTANTI__;   // costanti del fold-in, iniettate dal server

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// Forza del segnale: la norma di v_j decide quanto l'articolo sposta il
// vettore utente. Le soglie separano gli articoli quasi invisibili al modello
// (letti da pochissimi) da quelli con una rappresentazione vera.
function forza(v) {
  const n = v.norma < 1.5 ? 1 : (v.norma < 4.5 ? 2 : 3);
  const pallini = '●'.repeat(n) + '○'.repeat(3 - n);
  const quanti = v.click === null ? '' : ` ${v.click} click`;
  return `<span class="forza" title="norma del vettore latente: ${v.norma}${quanti}">`
       + `<span class="pieno">${pallini.slice(0, n)}</span>`
       + `<span class="vuoto">${pallini.slice(n)}</span></span>`;
}

function etichette(v) {
  return `${forza(v)}<span class="cat">${esc(v.categoria)}</span>
          <span class="cat">${esc(v.sottocategoria)}</span> ${esc(v.id)}`;
}

// Peso del profilo nel sistema del fold-in, calcolato come lo calcolerebbe il
// server: traccia della correzione diviso traccia del termine comune. La
// traccia di Σ v v^T e' la somma dei quadrati delle norme, quindi bastano i
// dati gia' presenti nelle schede.
function pesoProfilo() {
  const somma = profilo.reduce((acc, v) => acc + v.norma * v.norma, 0);
  return K.fattore * somma / K.traccia;
}

// L'avviso guarda il peso reale, non il numero di articoli: tre articoli letti
// da una persona sola pesano meno di uno molto letto, e il conteggio da solo
// darebbe un falso via libera.
function aggiornaAvviso() {
  const av = document.getElementById('avviso');
  const peso = pesoProfilo();
  if (profilo.length > 0 && peso < K.soglia) {
    av.innerHTML = `<b>Profilo debole: pesa il ${(peso * 100).toFixed(2)}% del sistema.</b>
      Nel fold-in la correzione portata dal profilo va confrontata con il termine comune
      w₀·VᵀV + λI: sotto il ${(K.soglia * 100).toFixed(0)}% il vettore utente resta vicino al
      prior, i punteggi sono bassi e il ranking si appoggia agli articoli più co-cliccati.
      Non conta quanti articoli scegliete, ma quanto sono letti: quelli con un solo pallino
      hanno un vettore latente quasi nullo e non spostano nulla.`;
    av.style.display = 'block';
  } else {
    av.style.display = 'none';
  }
}

function disegnaProfilo() {
  document.getElementById('conta').textContent = profilo.length;
  aggiornaAvviso();
  const box = document.getElementById('profilo');
  if (!profilo.length) { box.innerHTML = '<div class="vuoto">Nessun articolo scelto.</div>'; return; }
  box.innerHTML = profilo.map((v, i) => `
    <div class="voce">
      <div class="testo">
        <div class="titolo">${esc(v.titolo)}</div>
        <div class="etichette">${etichette(v)}</div>
      </div>
      <button onclick="togli(${i})">togli</button>
    </div>`).join('');
}

function aggiungi(v) {
  if (!profilo.some(p => p.id === v.id)) { profilo.push(v); disegnaProfilo(); }
}
function togli(i) { profilo.splice(i, 1); disegnaProfilo(); }
function svuota() {
  profilo = []; disegnaProfilo();
  document.getElementById('ranking').innerHTML = '<div class="vuoto">In attesa di un profilo.</div>';
  document.getElementById('tempi').textContent = '';
  document.getElementById('statoUtente').innerHTML = '';
}

let attesa;
document.getElementById('ricerca').addEventListener('input', e => {
  clearTimeout(attesa);
  const q = e.target.value;
  attesa = setTimeout(async () => {
    const box = document.getElementById('risultati');
    if (q.trim().length < 2) { box.innerHTML = ''; return; }
    const r = await fetch('/api/cerca?q=' + encodeURIComponent(q));
    const voci = await r.json();
    box.innerHTML = voci.length ? voci.map(v => `
      <div class="voce">
        <div class="testo">
          <div class="titolo">${esc(v.titolo)}</div>
          <div class="etichette">${etichette(v)}</div>
        </div>
        <button onclick='aggiungi(${JSON.stringify(v)})'>aggiungi</button>
      </div>`).join('') : '<div class="vuoto">Nessun articolo trovato.</div>';
  }, 160);
});

async function caricaUtente() {
  const id = document.getElementById('utente').value.trim();
  const stato = document.getElementById('statoUtente');
  if (!id) return;
  const r = await fetch('/api/utente?id=' + encodeURIComponent(id));
  const d = await r.json();
  if (!d.trovato) { stato.innerHTML = '<div class="errore">Utente non trovato.</div>'; return; }
  profilo = d.articoli;
  disegnaProfilo();
  const badge = d.nel_training
    ? '<span class="badge">presente nel training</span>'
    : '<span class="badge attenzione">NON presente nel training</span>';
  stato.innerHTML = `<div class="nota">${badge}
    profilo letto da data/${esc(d.split)}/behaviors.tsv — ${d.articoli.length} articoli
    utilizzabili${d.scartati ? ', ' + d.scartati + ' senza rappresentazione' : ''}.</div>`;
}

async function raccomanda() {
  const box = document.getElementById('ranking');
  if (!profilo.length) { box.innerHTML = '<div class="errore">Scegli almeno un articolo.</div>'; return; }
  box.innerHTML = '<div class="vuoto">Calcolo…</div>';
  const r = await fetch('/api/raccomanda', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({news: profilo.map(v => v.id),
                          quanti: +document.getElementById('quanti').value})
  });
  const d = await r.json();
  if (d.errore) { box.innerHTML = `<div class="errore">${esc(d.errore)}</div>`; return; }
  box.innerHTML = d.ranking.map((v, i) => `
    <div class="voce">
      <div class="rango">${i + 1}.</div>
      <div class="punti">${v.punteggio.toFixed(3)}</div>
      <div class="testo">
        <div class="titolo">${esc(v.titolo)}</div>
        <div class="etichette">${etichette(v)}</div>
      </div>
    </div>`).join('');
  document.getElementById('tempi').textContent =
    `fold-in e punteggio su ${d.catalogo.toLocaleString('it')} articoli in ${d.millisecondi} ms` +
    ` — esclusi i ${d.profilo.length} gia' nel profilo` +
    (pesoProfilo() < K.soglia ? ' · profilo debole: la scala dei punteggi misura la'
                              + ' confidenza, non la qualità del match' : '');
}
</script>
</body>
</html>
"""


class Gestore(BaseHTTPRequestHandler):
    catalogo: Catalogo = None

    def _invia(self, corpo: bytes, tipo: str, codice: int = 200):
        self.send_response(codice)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def _json(self, dato, codice: int = 200):
        self._invia(json.dumps(dato).encode("utf-8"), "application/json; charset=utf-8", codice)

    def do_GET(self):
        percorso = urlparse(self.path)
        parametri = parse_qs(percorso.query)

        if percorso.path in ("/", "/index.html"):
            costanti = json.dumps({
                "traccia": self.catalogo.traccia_comune,
                "fattore": self.catalogo.fattore_correzione,
                "soglia": SOGLIA_PROFILO_DEBOLE,
            })
            pagina = PAGINA.replace("__COSTANTI__", costanti)
            self._invia(pagina.encode("utf-8"), "text/html; charset=utf-8")
        elif percorso.path == "/api/cerca":
            self._json(self.catalogo.cerca(parametri.get("q", [""])[0]))
        elif percorso.path == "/api/utente":
            self._json(self._utente(parametri.get("id", [""])[0]))
        else:
            self._invia(b"non trovato", "text/plain; charset=utf-8", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/raccomanda":
            self._invia(b"non trovato", "text/plain; charset=utf-8", 404)
            return
        lunghezza = int(self.headers.get("Content-Length", 0))
        richiesta = json.loads(self.rfile.read(lunghezza) or b"{}")
        quanti = max(1, min(50, int(richiesta.get("quanti", 10))))
        self._json(self.catalogo.raccomanda(richiesta.get("news", []), quanti))

    def _utente(self, user_id: str) -> dict:
        dato = self.catalogo.profilo_utente(user_id.strip())
        if not dato.get("trovato"):
            return {"trovato": False}
        # Gli articoli senza colonna in V non sono utilizzabili per il fold-in.
        noti = [n for n in dato["news"] if n in self.catalogo.mappings.news2idx]
        return {"trovato": True, "split": dato["split"], "nel_training": dato["nel_training"],
                "articoli": [self.catalogo.scheda(n) for n in noti],
                "scartati": len(dato["news"]) - len(noti)}

    def log_message(self, formato, *argomenti):
        pass          # niente log per richiesta: durante una demo sono rumore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--porta", type=int, default=8000)
    parser.add_argument("--modello", default="artifacts/wmf")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("caricamento del modello e del catalogo…")
    Gestore.catalogo = Catalogo(args.modello)
    print(f"pronto: {len(Gestore.catalogo.id_per_indice):,} articoli, "
          f"k={Gestore.catalogo.modello.cfg.k}")

    indirizzo = f"http://localhost:{args.porta}"
    server = ThreadingHTTPServer(("127.0.0.1", args.porta), Gestore)
    print(f"in ascolto su {indirizzo}  (Ctrl-C per fermare)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(indirizzo)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nfermato")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
