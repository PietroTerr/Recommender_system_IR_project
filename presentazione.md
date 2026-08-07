# Presentazione — 15 minuti

Progetto #9, Weighted Matrix Factorisation su MIND.

Ogni slide riporta **cosa mettere**, **quale immagine** e **cosa dire**. I
tempi sono calibrati sulla lunghezza effettiva del parlato (circa 145 parole
al minuto) e sommano a 14:10, lasciando quasi un minuto
di margine sui quindici. La slide 14 è l'unica in cui il tempo dichiarato
eccede il parlato: comprende le tre interazioni con il browser.

Regola generale: sulle slide vanno numeri e figure, non frasi. Il testo qui
sotto è parlato, non va copiato sulla slide.

---

## Slide 1 — Il problema (45 s)

**Sulla slide:** titolo, nome, e le tre righe del testo del progetto:
*build a Weighted MF · accept as input a set of liked documents · return a
ranking of documents*.

**Immagine:** nessuna.

**Cosa dire:**

> Il progetto chiede tre cose: costruire una Weighted Matrix Factorisation per
> ottenere gli embedding di utenti e documenti, accettare in ingresso un utente
> descritto come un insieme di documenti che gli piacciono, e restituire un
> ranking. Ho lavorato su MIND, il dataset di notizie di Microsoft:
> cinquantamila utenti, cinquantunmila articoli. Tutto è implementato da zero —
> WALS, le metriche — senza librerie di recommender system. Anticipo la
> conclusione perché è la parte interessante: il modello funziona e fa quello
> che deve, ma su questo dataset una fattorizzazione pura non batte una
> baseline di popolarità corretta, e capire perché è stata la parte più
> istruttiva.

---

## Slide 2 — Il dataset (40 s)

**Sulla slide:** la tabella delle statistiche verificate — 50.000 utenti,
156.965 impression di training, 51.282 articoli, densità 0,045%. Una riga in
rosso: *fonte ufficiale non accessibile dal luglio 2024*.

**Immagine:** eventualmente uno screenshot dell'errore
`409 PublicAccessNotPermitted`.

**Cosa dire:**

> Prima sorpresa, ancora prima di scrivere codice: il dataset non si scarica.
> Lo storage account Microsoft che ospita MIND rifiuta l'accesso anonimo dal
> luglio 2024, le segnalazioni aperte sul repository ufficiale non hanno avuto
> risposta, e la pagina di Azure Open Datasets punta ancora a quel link morto.
> Ho usato un mirror e ho verificato l'autenticità confrontando le statistiche
> con quelle pubblicate: cinquantamila utenti esatti, 156.965 impression,
> 51.282 articoli. Combaciano tutte. La densità della matrice è dello 0,045 per
> cento: in forma densa occuperebbe venti gigabyte, quindi tutto il progetto
> lavora in rappresentazione sparsa.

---

## Slide 3 — Due sorprese nei dati (60 s)

**Sulla slide:** due riquadri. *La History è costante per utente* — 50.000 su
50.000. *Train e dev condividono solo 5.943 utenti su 50.000* — l'88% delle
impression di test ha un utente mai visto.

**Immagine:** `artifacts/figure/coda_lunga.png`. La distribuzione del feedback
positivo: a sinistra quanti utenti hanno cliccato ciascun articolo, a destra
quanti articoli ha ciascun utente, entrambe in scala log-log.

**Cosa dire:**

> Due proprietà del dataset che non sono documentate e che ho scoperto
> guardando i dati. La prima: il campo History è identico in tutte le
> impression dello stesso utente, anche a giorni di distanza. Non è una
> cronologia che cresce, è un profilo statico. Conseguenza pratica: se contate
> le occorrenze di un articolo nella History state misurando quante volte
> quell'utente è comparso nel log, non quanto gli piace l'articolo. Cinque
> milioni e ottocentomila token si riducono a un milione e centoquarantottomila
> coppie distinte. La seconda: training e test hanno cinquantamila utenti
> ciascuno ma ne condividono solo 5.943. Per l'ottantotto per cento del test
> l'utente è sconosciuto — e questo trasforma un requisito del progetto
> nell'unico modo possibile di valutare il modello. Il grafico mostra perché il
> problema è difficile: l'uno per cento degli articoli raccoglie il trentasei
> per cento dei click.

---

## Slide 4 — Feedback a tre livelli (55 s)

**Sulla slide:** la tabella dei tre casi con i conteggi.

| | valore | peso | coppie |
|---|---|---|---|
| cliccato | 1 | w_pos | 1.148.447 |
| mostrato e non cliccato | 0 | w_neg | 4.746.537 |
| mai mostrato | 0 | w₀ | il resto |

**Immagine:** eventualmente lo schema della matrice C con tre tonalità di cella.

**Cosa dire:**

> Questa è l'idea attorno a cui ho costruito il progetto. Nella formulazione
> vista a lezione le celle sono di due tipi: osservate, con peso w, e non
> osservate, con peso w-zero. MIND però registra gli impression log, cioè dice
> anche quali articoli sono stati *mostrati* all'utente senza essere cliccati.
> Quello non è un dato mancante: è un "no". Ho quindi tre livelli invece di due,
> e il secondo pesa quattro volte il primo — quasi cinque milioni di coppie
> contro un milione e centoquarantottomila. Cinquantaduemilaottocentosettanta
> coppie compaiono in entrambi i gruppi, mostrate e ignorate in una sessione e
> cliccate in un'altra: in quel caso vince il positivo. La parametrizzazione
> contiene i casi della lezione come casi particolari: con w-zero uguale a zero
> ottengo la observed-only, con tutti i pesi uguali ottengo la SVD.

---

## Slide 5 — La funzione obiettivo (40 s)

**Sulla slide:** la formula, grande e da sola.

```
min  Σ w_ij (C_ij − u_i·v_j)²  +  w₀ Σ (u_i·v_j)²  +  λ(‖U‖² + ‖V‖²)
```

**Immagine:** nessuna.

**Cosa dire:**

> La funzione obiettivo è quella della lezione. Il primo termine dice che sulle
> celle osservate voglio avvicinarmi al valore vero, pesando ogni cella con la
> sua confidenza. Il secondo dice che sulle celle mai osservate voglio stare
> vicino a zero, ma con confidenza molto più bassa: se forzassi a zero tutto ciò
> che non ho visto starei facendo una SVD, e affermerei che ogni articolo che
> l'utente non ha letto non gli piace. Il terzo è la regolarizzazione. Non è
> convessa nelle due matrici insieme, ma lo è in ciascuna tenendo fissa l'altra:
> da qui l'alternanza.

---

## Slide 6 — Il trucco che rende praticabile WALS (60 s)

**Sulla slide:** le due formule, e sotto in grande: **da O(n·k²) a
O((|P|+|N|)·k² + k³)** — da 51.282 colonne a ~119 celle per utente.

```
A_i = w₀·VᵀV + (w_pos−w₀)·Σ_{j∈P_i} v_j v_jᵀ + (w_neg−w₀)·Σ_{j∈N_i} v_j v_jᵀ + λI
b_i = w_pos·Σ_{j∈P_i} v_j
```

**Immagine:** nessuna. Questa è la slide della lavagna: se arriva una domanda
tecnica, arriva qui.

**Cosa dire:**

> Questo è il cuore. Annullando il gradiente rispetto al vettore di un utente
> ottengo un sistema lineare k per k. Scritto in modo ingenuo, però, la matrice
> di quel sistema è una somma su tutte le cinquantunmila colonne — per ognuno
> dei cinquantamila utenti, a ogni iterazione: impraticabile. L'osservazione che
> salva tutto è che le celle mai osservate hanno tutte lo stesso peso. Posso
> quindi scrivere la somma come se fosse estesa a tutte le colonne con peso
> w-zero, e poi correggere solo sulle celle che ho davvero osservato. Il primo
> termine è V trasposta per V: è trentadue per trentadue e si calcola una volta
> sola per mezza iterazione. La correzione tocca in media ventitré celle
> positive e novantasei negative. Il costo per utente crolla, e un'iterazione
> completa su tutto il dataset dura sei secondi.

---

## Slide 7 — Convergenza (40 s)

**Sulla slide:** **perdita monotona non crescente su tutte le iterazioni**, da
865.943 a 579.763.

**Immagine:** `artifacts/wmf/convergenza.png`. La funzione obiettivo iterazione
per iterazione: scende ripida sulle prime due e poi si appiattisce.

**Cosa dire:**

> WALS garantisce che la perdita non cresca, perché ogni mezzo passo risolve
> esattamente il minimo rispetto a un blocco di variabili. Ho usato questa
> proprietà come test: se la perdita sale, ho sbagliato le equazioni normali.
> Non sale mai. Un dettaglio di cui vado fiero: calcolare la perdita
> richiederebbe di sommare su due miliardi e mezzo di celle, ma la somma dei
> quadrati delle predizioni su tutte le celle è la traccia del prodotto delle
> due Gramiane. Costa quanto una moltiplicazione trentadue per trentadue,
> quindi monitoro la convergenza a ogni iterazione senza pagare nulla.

---

## Slide 8 — Il fold-in (55 s)

**Sulla slide:** *"The system must accept as input a user (a set of liked
documents)"* e sotto: **fold-in 0,2618 · riga di U appresa 0,2632 · differenza
0,5%**.

**Immagine:** uno schema: insieme di documenti → un solve k×k → vettore utente
→ punteggi su tutto il catalogo.

**Cosa dire:**

> Torniamo all'ottantotto per cento di utenti sconosciuti. Per loro non esiste
> una riga nella matrice U: l'unico modo di collocarli è partire dai documenti
> che gli piacciono. E qui c'è la cosa elegante: non serve un algoritmo nuovo. È
> esattamente un passo del ramo "fissa V e risolvi per u", ristretto alle celle
> positive — perché di un utente nuovo non so cosa gli sia stato mostrato senza
> essere cliccato. Un solo sistema k per k. Il requisito del progetto e il
> metodo coincidono. E l'ho verificato invece di darlo per buono: sui 5.943
> utenti presenti in entrambi gli split, ricavare il vettore dal profilo dà
> 0,2618 contro 0,2632 della riga appresa durante il training. Mezzo punto
> percentuale. Un utente descritto solo da quello che gli piace vale quanto un
> utente addestrato.

---

## Slide 9 — Come si valuta (50 s)

**Sulla slide:** due colonne. *Metriche*: MAP, P@k, R-precision, curva a 11
punti. *Protocolli*: per impression (37 candidati) e full-catalog (51.282
articoli).

**Immagine:** `artifacts/dev/curva_11_punti.png` — la curva di precisione
interpolata a 11 punti delle baseline.

**Cosa dire:**

> Sulla valutazione ho fatto una scelta consapevole. La letteratura su MIND usa
> AUC, MRR e nDCG; io uso le misure del corso. Non è una rinuncia: con rilevanza
> binaria misurano la stessa cosa. Nel settantuno per cento delle impression c'è
> un solo click, e in quel caso l'Average Precision *è* il reciproco del rango,
> cioè l'MRR; sul resto il MAP è più corretto perché tiene conto di tutti i
> click. L'nDCG serve per la rilevanza graduata, che qui non esiste. L'AUC l'ho
> comunque implementata, ma la uso solo come test: è l'unica di queste misure
> con un valore atteso noto, e uno scorer casuale mi dà 0,5007 su settantatremila
> impression — se non desse quello, il bug sarebbe nelle metriche, non nel
> modello.

---

## Slide 10 — Le baseline, e come si leggono i numeri (75 s)

**Sulla slide:** **uno scorer casuale ottiene già MAP = 0,2312**, e sotto:
popolarità 0,2423 · CTR smussato 0,2860.

**Immagine:** `artifacts/figure/map_modelli.png`. Le barre del MAP con la linea
tratteggiata del caso e, sopra ogni barra, il guadagno percentuale su di essa.

**Cosa dire:**

> Due cose insieme, perché sono la stessa cosa. Primo: il MAP non parte da zero.
> Con trentasette candidati per impression e in media un click e mezzo, uno
> scorer completamente casuale ottiene già 0,2312. Quindi quando fra poco
> vedrete 0,29 non state guardando un sistema che funziona al ventinove per
> cento: state guardando un più ventisei per cento rispetto al caso. La riga del
> random non è un vezzo, è il riferimento. Secondo: prima di costruire il
> modello ho costruito le baseline, e mi aspettavo che la popolarità fosse
> forte, come si legge un po' ovunque per le notizie. Non lo è: 0,2423 contro
> 0,2312, praticamente nulla. Quello che è forte è il click-through rate, cioè i
> click diviso le volte in cui l'articolo è stato mostrato: 0,2860. La
> differenza fra le due è il bias di esposizione. La popolarità grezza misura in
> larga parte cosa MSN ha deciso di mettere in homepage; il CTR normalizza per
> l'esposizione e isola la preferenza vera. Questa baseline, dieci righe di
> codice, è diventata l'avversario da battere.

---

## Slide 11 — Risultati (45 s)

**Sulla slide:** la tabella completa, con l'ibrido evidenziato.

| modello | MAP | R-prec |
|---|---|---|
| random | 0,2312 | 0,1005 |
| popolarità | 0,2423 | 0,1101 |
| CTR smussato | 0,2860 | 0,1551 |
| WMF | 0,2607 | 0,1291 |
| **ibrido WMF + CTR** | **0,2913** | **0,1556** |

**Immagine:** `artifacts/confronto_11_punti.png` — le curve a 11 punti di tutti
i modelli sovrapposte.

**Cosa dire:**

> Ecco il risultato, e non è quello che speravo. La fattorizzazione batte
> nettamente il caso, 0,2607 contro 0,2312, ma perde contro il CTR. E non è
> questione di taratura: ho fatto una ricerca su k, su w-zero, su w_neg e sulla
> regolarizzazione, e il MAP resta in un plateau fra 0,255 e 0,263. Però il
> segnale che la fattorizzazione cattura non è inutile: è *complementare*.
> Combinando i due punteggi, normalizzati dentro ogni impression, ottengo
> 0,2913, che batte il CTR su tutte e cinque le metriche. La fattorizzazione sa
> qualcosa che la popolarità non sa; semplicemente, da sola non basta.

---

## Slide 12 — La tesi verificata (50 s)

**Sulla slide:** i due estremi della curva — **w_neg = w₀ → MAP 0,2566 ·
w_neg = 1,0 → MAP 0,2637**, cioè **+2,8% di MAP e +6,2% di R-precision**, e la
parola *monotona*.

**Immagine:** `artifacts/figure/effetto_w_neg.png`. MAP (blu, asse sinistro) e
R-precision (arancione, asse destro) su sei valori di w_neg in scala
logaritmica. La linea verticale tratteggiata segna il punto in cui w_neg
coincide con w₀: a sinistra il modello *non* sta usando gli impression log.

**Cosa dire:**

> Questa è la verifica dell'ipotesi da cui ero partito, su sei valori invece che
> confrontandone due. Sulla sinistra il peso dei negativi osservati coincide con
> quello delle celle mai mostrate: lì il modello non distingue "gliel'ho
> mostrato e l'ha ignorato" da "non gliel'ho mai mostrato", ed è la Weighted MF
> classica a due livelli. Il MAP vale 0,2566. Spostandomi a destra sale in modo
> monotono fino a 0,2637, e la R-precision guadagna il sei per cento. Sei punti,
> nessuna inversione, due metriche che concordano: non è rumore. Una
> precisazione onesta: la curva sta ancora salendo all'estremo destro, quindi
> l'ottimo non l'ho trovato — ho verificato la direzione dell'effetto, non il
> suo massimo.

---

## Slide 13 — Cosa hanno imparato i 32 numeri (70 s)

**Sulla slide:** **47,2% delle raccomandazioni nella stessa sottocategoria del
profilo, contro 6,3% del caso — 7,4×**. E in piccolo: *category e subcategory
sono colonne di news.tsv che il modello non legge mai*.

**Immagine:** due figure affiancate. A sinistra
`artifacts/figure/specializzazione.png` (barre impilate: dove finiscono le
prime dieci raccomandazioni, per WMF, CTR e caso). A destra
`artifacts/figure/categorie.png` (similarità coseno fra i centroidi delle
categorie nello spazio latente: `tv` e `music` a 0,94, `finance` e `travel` a
0,75, `sports` isolata da tutto).

**Cosa dire:**

> Le metriche aggregate non dicono cosa il modello abbia capito. Queste due
> figure sì. A destra, la similarità fra le categorie nello spazio dei fattori:
> televisione e musica stanno a 0,94, mentre lo sport è isolato da tutto il
> resto. A sinistra la verifica quantitativa. In MIND ogni articolo ha due
> etichette, una categoria come "sports" e una sottocategoria come
> "football_nfl". Ho preso i duemilacento utenti il cui profilo è concentrato
> per almeno metà in una sola sottocategoria e ho guardato dove finiscono le
> prime dieci raccomandazioni: il quarantasette per cento cade nella stessa
> sottocategoria, contro il sei per cento che darebbe una scelta a caso. Sette
> volte tanto. E il confronto che chiude il discorso: il CTR, che non è
> personalizzato, si ferma all'undici per cento. Il punto è che il modello quelle
> etichette non le legge mai — la fattorizzazione vede solo la matrice di
> feedback. Che "football_nfl" sia una cosa diversa da "basketball_nba" è emerso
> dai soli pattern di co-click.

---

## Slide 14 — Demo dal vivo (110 s)

**Sulla slide:** niente, o solo il comando `python demo_web.py`. Passate al
browser a schermo intero.

**Immagine:** nessuna — è la demo.

**Come condurla.** Tre atti, venti secondi l'uno. Il server va **avviato prima
di cominciare la presentazione**: il caricamento del modello e dei titoli
richiede una quindicina di secondi che non volete spendere davanti a tutti.

*Atto 1 — un utente che non esiste.* Cercate `LeBron` e aggiungete **il solo
primo risultato**. Compare il riquadro giallo con la percentuale. Premete
*Raccomanda*.

> Sto costruendo un utente che nel dataset non esiste: gli piace questo
> articolo, punto. Il sistema lo colloca comunque, ma guardate i punteggi:
> due decimi, e metà delle raccomandazioni è fuori tema. Il modello mi sta
> dicendo che non sa, e mi dice anche quanto: l'avviso riporta che questo
> profilo pesa l'uno e tre per cento del sistema. È la formula di prima letta
> al contrario — la correzione portata dal profilo contro il termine comune
> w₀·VᵀV: finché la prima è trascurabile, il vettore utente resta dov'era.

*Atto 2 — il profilo si irrobustisce.* Aggiungete il secondo e il terzo
risultato. **Il riquadro sparisce.** Ripremete *Raccomanda*.

> Tre articoli, il peso è passato dall'uno al tre per cento e l'avviso è
> sparito. I punteggi sono saliti da 0,24 a 0,54 e i risultati in tema da
> cinque su dieci a nove su dieci. Non ho cambiato niente nel modello: ho solo
> dato più massa al termine di correzione. E il conto è durato due millisecondi
> in entrambi i casi.
>
> Un dettaglio: i risultati della ricerca sono ordinati per norma del vettore
> latente, e i pallini accanto a ciascuno la mostrano. Un articolo letto da una
> persona sola ha un vettore quasi nullo — la regolarizzazione lo ha schiacciato
> perché non c'è evidenza su cui stimarlo — e aggiungerlo al profilo non
> sposterebbe nulla.

*Atto 3 — un utente reale mai visto.* Nella casella dell'utente scrivete
`U90227` e premete *Carica*.

> Questo è un utente vero del test set, e il badge dice che **non è presente
> nel training**: non ha una riga nella matrice U. Sette articoli fra sport e
> vicende giudiziarie. Il sistema restituisce articoli su atleti coinvolti in
> casi penali — ha incrociato i due temi, non ha solo indovinato la categoria.
> Questo è l'ottantotto per cento del test set, ed è il requisito del progetto
> in funzione.

---

## Slide 15 — Conclusioni e limiti (55 s)

**Sulla slide:** tre righe di conclusione e tre di limiti dichiarati.

**Immagine:** nessuna.

**Cosa dire:**

> Chiudo con quello che ho imparato e con quello che non torna. I tre requisiti
> sono soddisfatti, e il fold-in — che sembrava un dettaglio — si è rivelato la
> parte centrale. Il feedback a tre livelli funziona. Una fattorizzazione pura
> però perde contro una baseline di popolarità corretta per l'esposizione, e
> credo di sapere perché: il protocollo ufficiale di MIND chiede di riordinare
> trentasette candidati che MSN ha già selezionato per tema e freschezza, dove
> la coerenza tematica conta poco perché ce l'hanno già tutti. I limiti che
> dichiaro: i valori di Recall sul catalogo completo sono piccoli in assoluto e
> non ho calcolato intervalli di confidenza; il venti per cento dei candidati di
> test sono articoli mai visti nel training e ricevono un punteggio medio; e i
> pesi dell'ibrido sono fissi, non tarati. Grazie.

---

## Note operative

- **Avviate `demo_web.py` prima di iniziare**, non davanti al pubblico: il
  caricamento del modello e dei 51.282 titoli dura una quindicina di secondi.
  Tenete la scheda del browser già aperta e pronta.
- **La demo va provata prima.** L'utente U90227 dell'atto 3 è verificato e dà
  l'output descritto. Per gli atti 1 e 2 basta prendere i primi tre risultati
  nell'ordine in cui escono: la ricerca è ordinata per forza del segnale, quindi
  in cima ci sono già gli articoli con un vettore latente vero. I numeri citati
  nel parlato (1,35% → 3,28%, da 5/10 a 9/10 in tema) sono quelli misurati sulla
  query `LeBron`; con un'altra query cambiano, quindi o usate quella o
  rimisurate.
- Se preferite il terminale, `python recommend.py --utente U19351 --top 5`
  resta valido: quattordici articoli, undici NFL, dieci raccomandazioni NFL su
  dieci. È il piano B se il browser fa i capricci.
- **Slide 6 è quella su cui arriveranno le domande.** Sapete riderivare il
  sistema lineare dalla funzione obiettivo della slide 5? Se sì, il resto scorre.
- **Se siete lunghi**, accorciate la slide 9 alle sole due righe sul perché MAP
  e non nDCG, e la 15 ai soli limiti.
- **Sull'onestà dell'esempio della demo:** U19351 restituisce dieci
  raccomandazioni NFL su dieci, ma è un caso favorevole. La media vera è il 47%
  della slide 13, e un utente pescato a caso ne dà tipicamente quattro o cinque
  su dieci. Se il professore prova dal vivo, meglio che lo abbiate detto voi
  prima. I profili delle sottocategorie larghe, come `newspolitics`, vanno
  peggio della media perché sconfinano in `newsus` e `newsworld`.
- **Domande probabili:** perché non convessa congiuntamente; cosa succede con
  w₀ = 0; perché la History non va usata come conteggio; perché il fold-in non
  usa i negativi; come si comporta il sistema su un utente senza storia.
- **Se chiedono dell'ottimo di w_neg** (slide 12, la curva sale ancora al
  bordo): l'esperimento verificava il *segno* dell'effetto, non il massimo, e
  oltre w_neg = 1 si peserebbe un articolo ignorato più di uno cliccato, il che
  richiede una giustificazione che non ho.
- **Se chiedono come sapete che il codice è corretto:** `test_correttezza.py`,
  e in particolare `test_gramiana_ramo_utenti`, che confronta la forma
  ottimizzata della slide 6 con la somma su tutte le colonne calcolata per
  esteso. Ho verificato che il test abbia denti innestando tre bug realistici:
  li prende tutti e tre, con quattordici ordini di grandezza fra "corretto" e
  il più sottile di essi.
- `pca_fattori.png` **non va in presentazione**: due componenti conservano solo
  l'11,7% della varianza e le categorie si sovrappongono. Resta nel repository
  come verifica fatta, non come risultato. Se qualcuno chiede una
  visualizzazione dello spazio latente, la risposta è la heatmap della slide 13.
