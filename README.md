# IR_recommender_project

Sistema di raccomandazione con Weighted Matrix Factorisation — Progetto #9 del
corso di Information Retrieval, Università di Trieste.

Dato un utente descritto come *insieme di documenti graditi*, il sistema
restituisce un ranking di articoli di notizie. Gli embedding di utenti e
articoli sono ottenuti con una **Weighted Matrix Factorisation** risolta
tramite **Weighted Alternating Least Squares**, implementata da zero: il
progetto non usa librerie di recommender system, e anche le metriche di
ranking sono scritte a mano.

Dataset: **MIND-small** (Microsoft News Dataset), 50.000 utenti e 51.282
articoli.

## Requisiti ed esecuzione

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Gli script vanno eseguiti **dalla radice del progetto**, in quest'ordine:

| comando | cosa produce | tempo |
|---|---|---|
| `python data_downloader.py` | `data/train/`, `data/dev/` (~210 MB) | qualche minuto |
| `python data_loader.py` | `artifacts/train/` — matrici di feedback e indice | ~2 min |
| `python evaluation.py` | struttura di valutazione, baseline, curva a 11 punti | ~3 min |
| `python wals.py` | `artifacts/wmf/` — modello addestrato e valutato | ~4 min |
| `python confronto.py` | confronto fra tutti i modelli e figura finale | ~3 min |
| `python figure.py` | figure per la relazione in `artifacts/figure/` | ~12 min |
| `python specializzazione.py` | analisi categoria/sottocategoria e figura | ~2 min |
| `python demo_web.py` | **demo**: interfaccia web su `localhost:8000` | immediato |
| `python recommend.py` | demo da riga di comando | immediato |

Ogni script salva i propri risultati su disco e li ricarica se già presenti,
quindi non c'è mai bisogno di rifare da capo un passaggio già eseguito.
`figure.py` è lento solo la prima volta, perché addestra sei modelli per la
curva su `w_neg`; le esecuzioni successive riusano la cache.

## Verifica

```bash
python test_correttezza.py --dataset
```

Dodici test in pochi secondi. Il rischio di questo progetto non sono i crash,
sono i numeri sbagliati ma plausibili: un MAP di 0,26 sembra ragionevole sia
che il codice sia giusto sia che si stia sbagliando un segno. I test
verificano quindi le **identità matematiche** su cui poggia l'implementazione,
confrontando ogni forma ottimizzata con la definizione calcolata per esteso su
matrici giocattolo:

- la riscrittura con la Gramiana, su entrambi i rami dell'alternanza, contro
  la somma su tutte le colonne;
- la perdita in forma chiusa contro la somma densa cella per cella;
- il fold-in che riproduce **esattamente** il passo di training, il che rende
  il requisito del progetto un'identità verificata e non una stima;
- perdita non crescente, determinismo, round-trip su disco, ranking che
  esclude il profilo.

Con `--dataset` si aggiungono i conteggi sugli artefatti reali (1.148.447
positivi, 4.746.537 negativi, disgiunzione, 73.152 impression di test): quei
valori provengono da un conteggio indipendente fatto con `awk` sui `.tsv`, non
dal codice, quindi intercettano un cambiamento di comportamento del parsing.

I test sono stati validati innestando tre bug realistici nella riscrittura
della Gramiana — segno invertito, `w₀` non applicato, negativi ignorati: li
intercetta tutti, con quattordici ordini di grandezza fra il caso corretto
(10⁻¹⁵) e il più sottile dei tre (10⁻¹).

## Demo

### Interfaccia web

```bash
python demo_web.py
```

Si apre su `http://localhost:8000`. Si compone un utente cercando gli articoli
per titolo e aggiungendoli al profilo, oppure si carica il profilo di un
utente reale del dataset; il sistema restituisce il ranking sull'intero
catalogo con categoria, sottocategoria e punteggio, e il tempo impiegato.

Usa solo `http.server` della libreria standard, con HTML e JavaScript in
linea: nessuna dipendenza aggiuntiva e nessuna risorsa esterna, quindi
funziona anche senza rete. Ascolta solo su `127.0.0.1`.

I risultati della ricerca sono ordinati per **forza del segnale**, cioè per
`‖v_j‖`, e ciascuno mostra tre pallini che la riassumono. La ragione è che un
articolo letto da pochissimi utenti viene schiacciato a zero dalla
regolarizzazione: aggiungerlo al profilo non sposta il vettore utente. Filtro e
ordinamento restano due passaggi distinti, quindi l'ordinamento non può far
entrare articoli che non corrispondono alla ricerca.

Quando il profilo è troppo debole compare un avviso. Il criterio non è il
numero di articoli ma il loro peso effettivo nel sistema del fold-in, cioè il
rapporto fra la traccia della correzione e quella del termine comune
`w₀·VᵀV + λI` (si sfrutta il fatto che `traccia(Σ v vᵀ) = Σ‖v‖²`, quindi il
calcolo è immediato lato pagina). Il conteggio da solo ingannerebbe:

| profilo | peso | risultati in tema |
|---|---|---|
| 3 articoli letti da una persona sola | 0,10% | rumore |
| 1 articolo molto letto | 1,35% | 5 su 10 |
| 3 articoli molto letti | 3,28% | 9 su 10 |
| 6 articoli molto letti | 4,39% | 9 su 10, satura |

La soglia è fissata al 2%, nel salto fra la seconda e la terza riga. Sotto di
essa il vettore utente resta vicino al prior, i punteggi calano di un ordine di
grandezza e il ranking si appoggia agli articoli più co-cliccati: la scala del
punteggio misura la confidenza, non la qualità del match.

### Riga di comando

```bash
python recommend.py --utente U90227 --top 5
```

```
modello: Weighted MF k=32  (51,282 articoli, 50,000 utenti)
utente:  U90227  —  NON presente fra i 50,000 del training

Documenti graditi in ingresso (7 utilizzabili):
  [tv     ] Jussie Smollett fails to persuade judge to drop Chicago lawsuit
  [sports ] Potential trade targets for all 32 NFL teams at the 2019 deadline
  [tv     ] John Witherspoon Dies: Comedian & 'Friday' Star Was 77
  [sports ] Report: Bengals bench QB Andy Dalton on his birthday
  ...

Ranking dei documenti (primi 5 su 51,282, esclusi quelli già nel profilo):
   1.  0.732  [sports] Charles Rogers, former Michigan State football, Detroit Lions
   2.  0.730  [sports] Former NFL lineman Justin Bannan arrested for attempted murder
   3.  0.685  [sports] Convicted ex-NFL tight end Kellen Winslow II has CTE symptoms
   4.  0.613  [sports] Former NBA first-round pick Jim Farmer arrested in sex sting
   5.  0.585  [sports] Frustrated Antonio Brown has active morning on Twitter

prodotto in 1.4 ms (fold-in + punteggio su tutto il catalogo)
```

L'utente **non esiste nel training**: viene collocato nello spazio latente
partendo dai soli sette articoli che gli piacciono. Il profilo unisce sport e
vicende giudiziarie, e il ranking restituisce articoli su **atleti coinvolti
in casi penali**: non è coincidenza di categoria, è struttura latente.

Altre modalità:

```bash
python recommend.py --news N55189 N42782 N34694
python recommend.py                       # utente a caso dal dev
```

## Struttura

```
data_downloader.py    scarica MIND-small
data_loader.py        parsing dei .tsv, costruzione di C_pos e M_neg, indice su disco
evaluation.py         metriche di ranking, baseline, protocolli di valutazione
wals.py               WALS: alternanza, fold-in, salvataggio del modello
confronto.py          confronto fra WMF, baseline e modello ibrido
demo_web.py           interfaccia web locale per la demo
recommend.py          demo da riga di comando
figure.py             figure per la relazione
specializzazione.py   quanto è specifica una raccomandazione
test_correttezza.py   test delle identità matematiche e degli invarianti
presentazione.md      traccia della presentazione, 15 slide
```

## Il dataset

**La fonte ufficiale non è più accessibile.** Dal luglio 2024 lo storage
account Microsoft che ospita MIND rifiuta l'accesso anonimo con HTTP 409
`PublicAccessNotPermitted`; le segnalazioni aperte sul repository del dataset
e sulla libreria `recommenders` non hanno avuto risposta, e la pagina Azure
Open Datasets punta ancora a quel percorso. Si usa quindi un mirror su
Hugging Face che espone i file `.tsv` originali senza modifiche.

L'autenticità è stata verificata confrontando formato e statistiche con
quelle pubblicate:

| | valore |
|---|---|
| utenti distinti (train) | 50.000 |
| impression train / dev | 156.965 / 73.152 |
| articoli train / dev | 51.282 / 42.416 |

### Due proprietà non ovvie, verificate sui dati

**La `History` è costante per utente.** Per tutti i 50.000 utenti la stringa
`History` è identica in ogni loro impression, anche a giorni di distanza: è un
profilo statico di click precedenti alla finestra del dataset, non una
cronologia cumulativa. Conseguenza pratica: contare le occorrenze grezze di un
articolo nella `History` misurerebbe *quante impression ha avuto l'utente*,
non l'intensità della preferenza. Le coppie (utente, articolo) vanno
deduplicate: 5,8 milioni di token diventano 1.148.447 coppie distinte.

**Train e dev contengono utenti diversi.** Hanno 50.000 utenti ciascuno, ma ne
condividono solo **5.943**. Per l'88% delle impression del dev non esiste una
riga in `U`, e l'unico modo di ottenere un punteggio personalizzato è ricavare
il vettore utente dalla sua `History`. Il fold-in richiesto dal progetto non è
quindi una funzionalità accessoria: è l'unico modo di valutare il modello.

## Modello

### Feedback a tre livelli

Gli impression log di MIND dicono anche cosa è stato *mostrato senza essere
cliccato*, il che permette di distinguere tre casi invece dei soliti due:

| caso | C_ij | peso | coppie |
|---|---|---|---|
| cliccato (o in `History`) | 1 | `w_pos` | 1.148.447 |
| mostrato e non cliccato | 0 | `w_neg` | 4.746.537 |
| mai mostrato | 0 | `w₀` | il resto |

Le 52.870 coppie che compaiono in entrambi i primi due gruppi (mostrate e
ignorate in una sessione, cliccate in un'altra) sono assegnate ai positivi.
La parametrizzazione include i casi discussi a lezione: `w₀ = 0` dà la
observed-only MF, `w₀ = w_pos = w_neg` dà la SVD.

### WALS e la riscrittura con la Gramiana

Annullando il gradiente rispetto a `u_i` si ottiene un sistema `k × k` la cui
matrice, scritta in modo ingenuo, somma su tutte le 51.282 colonne. Poiché le
celle non osservate condividono lo stesso peso, la somma si spezza in un
termine comune più una correzione sulle sole celle osservate:

```
A_i = w₀·VᵀV + (w_pos−w₀)·Σ_{j∈P_i} v_j v_jᵀ + (w_neg−w₀)·Σ_{j∈N_i} v_j v_jᵀ + λI
b_i = w_pos·Σ_{j∈P_i} v_j
```

`VᵀV` è `k × k` e si calcola una volta per mezza iterazione; la correzione
tocca solo le ~23 celle positive e ~96 negative dell'utente. Il costo per
utente passa da `O(n·k²)` a `O((|P_i|+|N_i|)·k² + k³)`, e un'iterazione
completa impiega circa 6 secondi. La perdita è calcolata in forma esatta senza
espandere la matrice densa, sfruttando `Σ_ij (u_i·v_j)² =
traccia((UᵀU)(VᵀV))`, e si verifica che sia non crescente a ogni iterazione.

Il **fold-in** è lo stesso sistema ristretto alle celle positive: di un utente
nuovo non si sa cosa gli sia stato mostrato senza essere cliccato.

## Valutazione

Le metriche sono quelle del corso — **MAP, P@k, R-precision** e la curva di
**precisione interpolata a 11 punti** — e non AUC/MRR/nDCG usate in
letteratura su MIND. Con rilevanza binaria le due famiglie misurano la stessa
cosa: nel 71,2% delle impression del dev c'è un solo click, e lì l'Average
Precision coincide con il reciproco del rango. L'AUC resta implementata ma
serve solo come test di correttezza, essendo l'unica misura con valore atteso
noto su uno scorer casuale (verificato: 0,5007 su 73.152 impression).

**Il MAP non ha pavimento a zero.** Con ~37 candidati per impression e in
media 1,5 click, uno scorer casuale ottiene MAP 0,2312: la riga `random` va
sempre riportata, altrimenti i numeri non sono interpretabili.

Sono usati due protocolli, e danno risposte diverse:

- **per impression** — si riordinano i ~37 candidati mostrati da MSN
  (protocollo ufficiale di MIND);
- **full-catalog** — si sceglie fra tutti i 51.282 articoli, escludendo quelli
  già letti. È il compito che il testo del progetto descrive.

## Risultati

### Protocollo per impression

| modello | MAP | P@5 | P@10 | R-prec | 11-point |
|---|---|---|---|---|---|
| random | 0,2312 | 0,1004 | 0,1002 | 0,1005 | 0,2353 |
| popolarità | 0,2423 | 0,1014 | 0,1030 | 0,1101 | 0,2465 |
| CTR smussato | 0,2860 | 0,1201 | 0,1103 | 0,1551 | 0,2905 |
| WMF | 0,2607 | 0,1112 | 0,1063 | 0,1291 | 0,2652 |
| **ibrido (WMF + CTR)** | **0,2913** | **0,1238** | **0,1128** | **0,1556** | **0,2958** |

### Protocollo full-catalog (2.000 utenti)

| modello | Recall@10 | Recall@50 | Recall@100 |
|---|---|---|---|
| random | 0,0000 | 0,0001 | 0,0015 |
| popolarità | 0,0000 | 0,0007 | 0,0007 |
| CTR smussato | 0,0016 | 0,0019 | 0,0083 |
| WMF | 0,0001 | 0,0020 | 0,0033 |
| ibrido (WMF + CTR) | 0,0014 | 0,0031 | 0,0081 |

## Cosa se ne ricava

**La popolarità grezza è quasi indistinguibile dal caso** (MAP 0,2423 contro
0,2312), mentre il CTR è forte. La popolarità misura in larga parte *cosa MSN
ha deciso di mostrare*; il CTR normalizza per l'esposizione e isola la
preferenza. È bias di esposizione allo stato puro.

**Il feedback a tre livelli funziona.** Portare `w_neg` da 0,025 — cioè
trattare "mostrato e ignorato" come "mai mostrato" — a 0,5 porta il MAP da
0,2574 a 0,2630 e la R-precision da 0,1249 a 0,1313: **+2,2% e +5,1%** solo
per aver usato l'informazione degli impression log. È l'ipotesi centrale del
progetto, e regge.

**Il fold-in non costa nulla in qualità.** Sui 5.943 utenti presenti in
entrambi gli split, ricavare il vettore dalla `History` dà MAP 0,2618 contro
0,2632 ottenuto usando la riga di `U` appresa: 0,5% di differenza. Un utente
descritto solo dai documenti che gli piacciono vale quanto un utente
addestrato — il che rende il requisito del progetto una proprietà misurata,
non un'asserzione.

**Una WMF pura perde contro il CTR sul protocollo per impression**, e nessuna
taratura di `k`, `w₀`, `w_neg` o `λ` ha colmato il divario (plateau fra 0,255
e 0,263). Il segnale collaborativo però è complementare, non inutile:
combinato con il CTR produce l'unico modello che batte la baseline su tutte e
cinque le metriche.

**Le raccomandazioni sono topicalmente coerenti anche quando le metriche non
lo premiano.** Su 300 utenti il modello produce 144 articoli distinti al primo
posto e 576 distinti nelle top-10: non ha collassato sulla popolarità. Il
protocollo per impression misura un compito diverso — riordinare 37 candidati
che MSN ha già selezionato per tema e freschezza — dove la coerenza topica
conta poco, perché ce l'hanno già tutti i candidati.

**Il modello specializza sulla sottocategoria, non sulla macro-categoria.**
Ogni articolo di MIND porta due etichette: una categoria (17 valori, es.
`sports`) e una sottocategoria (264 valori, es. `football_nfl`). Sui 2.126
utenti il cui profilo è concentrato per almeno metà in una sola
sottocategoria, ecco dove finiscono le prime dieci raccomandazioni:

| | stessa sottocat. | stessa cat. | altrove |
|---|---|---|---|
| **WMF** | **47,2%** | 16,3% | 36,5% |
| CTR | 11,6% | 18,4% | 70,1% |
| caso | 6,3% | 18,4% | 75,3% |

Il lift è **7,4× sulla sottocategoria** contro 2,6× sulla categoria: il
guadagno fine è quasi il triplo di quello grosso, quindi il modello non si
ferma a "sport". Il CTR, non essendo personalizzato, ha una quota di "stessa
categoria" identica al caso alla prima cifra decimale.

Il punto è che **il modello quelle etichette non le legge mai**: la
fattorizzazione vede solo la matrice di feedback, e `category` e `subcategory`
sono colonne di `news.tsv`. Che `football_nfl` sia una cosa diversa da
`basketball_nba` è emerso dai soli pattern di co-click; le etichette servono
qui unicamente da metro esterno. Lo stesso segnale si vede nella similarità
coseno fra i centroidi delle categorie nello spazio latente (`tv` e `music` a
0,94, `sports` isolata da tutto).

## Limiti noti e lavoro futuro

- I valori di **Recall@100 sono molto piccoli in assoluto** (0,001–0,008) e
  calcolati su un campione di 2.000 utenti. Le variazioni relative, anche
  grandi, non sono corredate da un intervallo di confidenza.
- **Le perdite non sono confrontabili fra configurazioni diverse**: ognuna
  minimizza una funzione obiettivo differente. La colonna della perdita
  certifica solo la monotonia dentro una singola esecuzione.
- Il **20% dei candidati del dev sono articoli assenti dal training** e
  ricevono il punteggio di un articolo medio. Un fallback content-based sui
  titoli (TF-IDF, modello vettoriale) chiuderebbe il buco, ma la diagnosi
  mostra che il divario con il CTR persiste anche sui soli candidati warm:
  non è il cold start la causa principale.
- L'**ibrido usa pesi fissi** (z-score sommati con peso 1). Tararne il
  rapporto sul dev è il primo esperimento da fare.
