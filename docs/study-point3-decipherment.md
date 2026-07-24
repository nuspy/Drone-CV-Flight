# Studio — Punto 3: decifrazione di una lingua ignota tramite vincoli tipologici

> Documento di design. Obiettivo: rendere operativo (e *falsificabile*) il passaggio
> dalla struttura di un testo ignoto a ipotesi ordinate sul suo significato.
>
> **Non** produce "la traduzione". Produce **ipotesi ordinate, con confidenza e
> corredate del proprio test di confutazione.**

---

## 0. La tesi che regge tutto

Il problema del punto 3 non è di potenza di calcolo, è di **informazione**. Un testo
puramente distribuzionale è *sottodeterminato nel significato*: puoi ricostruire tutta la
struttura (che ⟨A⟩ si comporta da articolo, ⟨B⟩ da verbo transitivo) ma non *quale* verbo
sia. Il significato non è nel testo; entra solo tramite un'**ancora** verso qualcosa di già
noto (bilingue, lingua parente, illustrazioni, numerali, nomi propri, contesto archeologico).

Da qui il rischio centrale, che è epistemico prima che tecnico:

> **Qualunque mapping abbastanza flessibile produce letture "plausibili".**

È il motivo per cui esistono decine di "Voynich risolto" mutuamente incompatibili. Un
generatore di ipotesi senza un generatore di *confutazioni* non è uno strumento di
decifrazione: è una macchina per illusioni coerenti.

Conseguenza di design: il punto 3 è composto da **due metà di pari peso**:

1. **Generazione** di ipotesi (struttura → tipologia → candidati → allineamento → lettura).
2. **Falsificazione** delle ipotesi (null model, predizione su held-out, controlli positivi,
   convergenza fra metodi, MDL).

La seconda metà è ciò che distingue questo progetto dalla pseudo-decifrazione. Nel codice
vive in `decipher/falsification/`.

---

## 1. L'input: cosa i punti 1–2 consegnano al punto 3

Il punto 3 non tocca i pixel. Riceve *strutture* già estratte:

| Dai punti 1–2 | Modulo | Uso nel punto 3 |
|---|---|---|
| Traslitterazione stabile (Unicode PUA) | `ocr.transliterate` | sequenza di simboli neutri, senza bias latino |
| Segmentazione in token/parole | `ocr` + `structure` | unità su cui indurre lessico |
| Fingerprint statistico (Zipf, entropia h₁…hₙ, type/token, lunghezze) | `structure.stats` | filtro "è lingua?"; indice di sintesi morfologica |
| Inventario morfologico (radici, affissi, paradigmi) | `structure.morphology` | separare lessico da grammatica |
| Ruoli distribuzionali (classi POS indotte, frame argomentale) | `structure.roles` | individuare verbi / soggetti / oggetti |

Questi sono i **vincoli**. Tutto il punto 3 consiste nel propagarli in avanti mantenendone
l'incertezza, senza mai spacciare un vincolo debole per una certezza.

---

## 2. Dalla struttura alla tipologia (primo salto — moderatamente affidabile)

Si mappano le feature *osservate* su feature *tipologiche standardizzate*
(WALS — World Atlas of Language Structures, ~192 feature × ~2.600 lingue; Grambank, ~195
feature × ~2.400 lingue). Feature ricavabili dai punti 1–2 senza conoscere il significato:

- **Ordine dominante S/V/O** — dalla role induction + posizione. (WALS 81A)
- **Tipo morfologico** (isolante / agglutinante / flessivo / polisintetico) — dall'*indice di
  sintesi* di Greenberg (morfemi per parola) e dall'entropia morfologica.
- **Head-directionality**, ordine Aggettivo–Nome, adposizioni pre/post — da co-occorrenza.
- **Marcatura di caso / accordo** — da alternanze di affissi correlate alla posizione.
- **Reduplicazione, incorporazione**, ecc. — da pattern ripetuti interni alla parola.

Prodotto: un **vettore tipologico con incertezze** (`hypothesis.typology.TypologicalFingerprint`).

> ⚠️ **Caveat forte, da non dimenticare mai: tipologia ≠ genealogia.**
> SOV + agglutinante descrive turco, giapponese, coreano, quechua, dravidico… lingue **non
> imparentate**. La tipologia **restringe** lo spazio delle lingue ma **non identifica** una
> famiglia. È un *prior*, non un match. Chi tratta la somiglianza tipologica come parentela
> produce falsi positivi a valanga.

---

## 3. Dalla tipologia ai candidati (salto debole — trattarlo come ranking probabilistico)

Si interroga un DB tipologico per le lingue col fingerprint compatibile, ordinandole per
distanza tipologica → **lista ordinata di candidati con punteggio**, mai un vincitore unico.

Il punto cruciale, e onesto: la tipologia da sola è troppo debole. I **prior extra-testuali
valgono più della tipologia pura** e vanno inseriti quando esistono:

- **geografia** del ritrovamento e datazione;
- **materiale e contesto** archeologico (moneta? lapide? contratto? testo religioso?);
- **sistema di scrittura** affine (un abjad? un sillabario? logografico?);
- lingue **attestate nella stessa area/epoca**.

Prodotto: una **distribuzione di probabilità** su famiglie/lingue candidate
(`hypothesis.candidates`). Design esplicitamente **multi-ipotesi**: si portano avanti in
parallelo i top-k candidati, non uno.

---

## 4. Dal candidato all'allineamento (dove si tenta di estrarre significato)

Quattro metodi, ciascuno con la **precondizione** che lo rende applicabile. Il sistema
tenta quelli le cui precondizioni sono soddisfatte e li fa **votare** (§6).

### (a) Ancore forti — sempre per prime
Elementi il cui *significato è vincolato dalla struttura interna*, non dalla lingua:

- **Numerali**: base 10/20/60, composizione additiva/moltiplicativa, sequenze ordinali sono
  riconoscibili senza conoscere la lingua. Storicamente il primo grimaldello.
- **Nomi propri, toponimi, teonimi**: bassa frequenza, alta ricorrenza in contesti fissi.
- **Date, unità di misura, formule ripetute** (colofoni, invocazioni, intestazioni).

Le ancore danno i primi punti fissi da cui propagare l'allineamento.

### (b) Cognate-based decipherment con lingua parente nota
La linea di ricerca su Ugaritico↔Ebraico e Lineare B↔Greco (allineamento neurale di
caratteri con *prior fonetico*, formulazioni tipo minimum-cost flow).
**Precondizione:** esiste un parente noto *e* lo script codifica fonologia recuperabile.
È il metodo che ha decifrazioni reali al suo attivo — ma solo quando la precondizione regge.

### (c) Unsupervised bilingual lexicon induction
Allineare lo spazio di embedding della lingua ignota a quello di una lingua candidata via
sola struttura distribuzionale (famiglia VecMap / MUSE), senza dizionario.
**Precondizione:** corpora comparabili di dominio simile e strutture distribuzionali
abbastanza isomorfe. **Fragile** con corpus piccolo e dominio ristretto — cioè il caso
tipico dei testi antichi. Va usato ma pesato basso quando il corpus è esiguo.

### (d) LLM come proponitore di letture — l'angolo moderno, con il guinzaglio corto
Dato il fingerprint + il lessico parziale indotto, si chiede a un LLM di produrre letture
grammaticali e coerenti sotto l'ipotesi corrente.
- **Utile** come generatore di ipotesi e come "riempitore" di lacune plausibili.
- **Pericolosissimo come giudice**: allucina coerenza, è progettato per farlo. Se lo lasci
  valutare la bontà della propria lettura, ti mente in modo convincente.
- Regola ferrea: l'LLM **propone**, la §6 **dispone**. Ogni lettura LLM passa per null model
  e held-out come tutte le altre. Mai chiudere il loop di validazione dentro l'LLM.

---

## 5. Scoring interno di un'ipotesi: coerenza e parsimonia (MDL)

Un'ipotesi = *(candidato + lessico indotto + regole grammaticali)*. Si misura:

- **copertura** del lessico sul corpus;
- **coerenza/grammaticalità** delle letture prodotte;
- **stabilità**: la stessa parola riceve lo stesso senso in contesti diversi?
- **parsimonia**: quante regole *ad hoc* / eccezioni servono?

Il criterio unificante è **MDL — Minimum Description Length**:

> La migliore decifrazione è quella che **comprime** meglio il testo:
> `L(grammatica) + L(lessico) + L(testo | grammatica, lessico)` minima.

Una decifrazione vera *comprime* (cattura regolarità reali). Una pseudo-decifrazione richiede
tante eccezioni quante ne "spiega" → non comprime → viene penalizzata automaticamente. MDL è
il rasoio di Occam reso una funzione di costo (`hypothesis.score`).

---

## 6. La macchina della falsificazione (il cuore)

Senza ground truth, l'unico modo di non ingannarsi è confrontarsi con controlli.

### 6.1 Null model
Applicare lo *stesso identico* pipeline a:
- (i) testo **shufflato** a livello di parola (distrugge sintassi, tiene le frequenze);
- (ii) testo **random** con la stessa distribuzione di frequenza (tiene solo Zipf);
- (iii) una **lingua nota mascherata** come controllo positivo.

Se l'ipotesi sul testo vero **non batte** i null (i) e (ii), il risultato è privo di
significato — punto. Questo test da solo demolisce la maggioranza delle pseudo-decifrazioni.
(`falsification.null_model`).

### 6.2 Predizione su held-out
Grammatica + lessico indotti su una parte del corpus devono **predire** la parte tenuta da
parte meglio di un baseline n-gram (perplexity più bassa). La struttura vera *generalizza*;
il sovradattamento no. (`falsification.heldout`).

### 6.3 Controllo positivo end-to-end
Passare una lingua *nota* (es. latino, o etrusco parzialmente noto) attraverso l'intero
pipeline trattandola come ignota, e misurare quanto si recupera. Calibra le aspettative e
funge da test di regressione del sistema. Se non recupera una lingua nota, non decifrerà una
ignota.

### 6.4 Convergenza fra metodi
I metodi (b), (c), (d) devono **convergere** sullo stesso lessico. Convergenza indipendente =
segnale; divergenza = rumore. Un accordo è credibile in proporzione all'indipendenza dei
metodi che lo producono.

### 6.5 Pre-registrazione
Fissare metriche e soglie **prima** di guardare l'output, per non fare cherry-picking a
posteriori.

---

## 7. Cosa il sistema può e non può concludere (matrice degli esiti)

| Caso | Ancora disponibile | Esito onesto |
|---|---|---|
| Lingua nota, script ignoto (es. Lineare B) | nomi, struttura fonologica | **lettura verificabile** probabile |
| Lingua ignota, parente noto (es. Ugaritico) | cognati | **lessico parziale** + confidenza |
| Isolato, con qualche ancora (numerali, bilingue frammentario) | ancore locali | **isole di significato**, resto strutturale |
| Isolato, nessuna ancora | nessuna | struttura completa, significato **no** |

Nell'ultimo caso il valore del sistema **non è zero**: caratterizza la struttura, **esclude**
famiglie candidate, e — soprattutto — **dice esattamente quale ancora servirebbe** per
sbloccare il resto. Dichiarare "significato non recuperabile con i dati attuali" è un
risultato scientifico, non un fallimento.

---

## 7-bis. Bypass del verdetto negativo (escape hatch, con avviso)

Il verdetto di falsificazione è di default un muro: se il corpus non è *language-like*, o se
nessuna ipotesi sopravvive, il pipeline si ferma. È giusto così — ma a volte l'utente vuole
comunque *vedere la migliore ipotesi*, sapendo che i numeri dicono che non è reale (per
esplorazione, per intuizione, per capire *cosa* fallisce).

Per questo `pipeline.run(..., override_feasibility=True)` **forza il passaggio** oltre il
verdetto negativo, con tre garanzie di onestà non negoziabili:

1. **Avviso rumoroso**: `warnings.warn` a runtime *e* la lista in `report.warnings`.
2. **Segregazione dei risultati**: gli output forzati vanno in `report.forced`, **mai** in
   `report.survivors`. Non si mescolano mai risultati forzati con quelli verificati —
   mescolarli spaccerebbe per verificato ciò che non lo è.
3. **Marchio esplicito**: `report.feasibility_overridden = True` e messaggio che dichiara
   "ipotesi forzate, nessuna garanzia statistica — NON presentare come decifrazione".

In sintesi: il bypass ti dà la miglior ipotesi *dopo* averti detto in faccia che non regge.
La responsabilità di non chiamarla "decifrazione" resta di chi legge — il sistema fa di tutto
per ricordarglielo.

## 8. Pipeline algoritmica (pseudocodice) → mappatura sui moduli

```
input:  translitterazione PUA + segmentazione        # punti 1–2
        prior extra-testuali (geo, epoca, materiale)  # opzionali ma preziosi

1. fingerprint      = structure.stats(text)                 # è lingua? indice sintesi
   if not is_language_like(fingerprint): STOP → report      # falsification-first
2. morph            = structure.morphology(text)
   roles            = structure.roles(text)
3. typo_vec         = hypothesis.typology(fingerprint, morph, roles)
4. candidates       = hypothesis.candidates(typo_vec, priors)      # ranking probabilistico
5. anchors          = hypothesis.align.find_anchors(text)          # numerali, nomi, formule
6. for cand in candidates.top_k():                                  # multi-ipotesi
       hyp = hypothesis.align(text, cand, anchors, methods=[b,c,d])
       hyp.score = hypothesis.score(hyp)                            # MDL + coerenza
7. for hyp in hypotheses:                                           # LA METÀ CHE CONTA
       hyp.null   = falsification.null_model(hyp)                   # batte shuffle/random?
       hyp.heldout= falsification.heldout(hyp)                      # generalizza?
8. survivors = [h for h in hypotheses if h.beats_nulls() and h.generalizes()]
   report(survivors ordered by confidence, + "quale ancora sbloccherebbe di più")
```

Orchestrazione in `decipher/pipeline.py`.

---

## 9. Limiti, rischi, responsabilità

- **Soglia di corpus**: sotto una certa mole non si distingue nemmeno lingua da rumore. La
  quantità di testo aiuta la *forma*, non il *contenuto*.
- **Rischio mediatico**: una "decifrazione" fa notizia; pubblicare *sempre* confidenza e
  risultato dei null model è un obbligo, non un optional.
- **Circolarità dell'LLM**: mai lasciargli chiudere il proprio loop di validazione.
- **Onestà terminale**: se i sopravvissuti alla §6 sono zero, il report lo dice. "Non
  decifrabile con questi dati" è l'output corretto, non un bug.

---

## 10. Riferimenti (da verificare — richiamati a memoria, non bibliografia definitiva)

- WALS — *World Atlas of Language Structures* (wals.info); Grambank (grambank.clld.org).
- Greenberg, indice di sintesi morfologica (quantitative typology).
- Rao et al., analisi di entropia condizionale sullo script dell'Indo (dibattito "è lingua?").
- Snyder, Barzilay, Knight — decifrazione dell'Ugaritico (2010).
- Luo, Cao, Barzilay — *Neural Decipherment via Minimum-Cost Flow* (2019); Luo et al.,
  decifrazione di script non segmentati con prior fonetico (2021).
- Artetxe et al. — VecMap; Lample et al. — MUSE (bilingual lexicon induction non supervisionato).
- Kondrak — identificazione di cognati e correspondenze fonetiche.
- Rissanen — Minimum Description Length.

> Nota: i riferimenti sopra sono richiamati dalla memoria del modello e vanno **verificati**
> (autori, anni, titoli esatti) prima di citarli in un lavoro formale.
