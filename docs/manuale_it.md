# Manuale End-to-End

*(English version: [manual_en.md](manual_en.md))*

Questo manuale guida dall'inizio alla fine entrambi gli usi del progetto:

- **Parte A — Generazione di ambienti 3D ad alta qualità** per Unity /
  Blender / visualizzazione (il "prodotto 3D").
- **Parte B — Test manuale di riconoscimento "da velivolo"**: foto aerea →
  coordinate geografiche, modalità streaming live, e il test di
  affidabilità automatico (il prodotto navigazione).

Le due parti condividono il primo passo: costruire un ambiente del mondo
reale dai dati GIS aperti.

---

## 0. Prerequisiti

| Cosa | Note |
|---|---|
| Python 3.11+ | `python --version` |
| Installazione | `git clone … && cd Drone-CV-Flight && pip install -e ".[dev,gis]"` (usa un venv) |
| GPU | opzionale — tutto gira su CPU; con CUDA il training è molto più veloce |
| Blender 3.6+/4.x | solo per l'export nativo `.blend` (deve stare nel `PATH`) |
| Unity 2022.3+ | solo per la scena Unity; installa il package **glTFast** (`com.unity.cloud.gltfast`) per i materiali |
| Rete | Overpass/OSM funziona su reti normali (`--source osm` negli script, provider di default in `gis build`). Su reti filtrate (proxy solo-AWS) usa i provider Overture — i dati arrivano da S3 pubblico |
| Disco | ~1 GB per ambiente a 1 m/px (2×2 km); i dataset di training aggiungono qualche centinaio di MB |

Tutti i comandi si lanciano dalla radice del repository, con il virtualenv
attivo (`dronecv` viene installato come comando da `pip install -e`).

---

## Parte A — Export 3D ad alta qualità, end to end

### A1. Scegliere l'area

GUI desktop (ricerca località, pan/zoom, disegno della mask poligonale
esatta, pannello copertura sorgenti):

```bash
dronecv gis gui
```

Oppure direttamente con un bounding box (lat1,lon1,lat2,lon2 = angolo SW,
angolo NE), dopo aver verificato che dati esistono lì:

```bash
dronecv gis info --bbox 47.4925,19.0290,47.5105,19.0560
```

`info` riporta la disponibilità delle tile DEM, il numero di edifici e la %
con altezze reali — decidi *prima* del build se l'area richiede la
ricostruzione da immagini satellitari.

### A2. Build alla massima qualità

```bash
dronecv gis build \
    --bbox 47.4925,19.0290,47.5105,19.0560 \
    --env-name budapest_hq \
    --res 1.0 \
    --reconstruct-buildings --imagery eox --imagery-res 10 \
    --palette-photos mie_foto/ \
    --ortho mia_ortofoto.tif --ortho-utc 2025-06-21T10:00:00Z
```

Ogni flag è una leva di qualità — usa ciò che hai, tutto è opzionale
tranne `--bbox`/`--place` e `--env-name`:

| Leva | Effetto | Quando usarla |
|---|---|---|
| `--res 1.0` | mosaico a 1 m/px: bordi degli edifici nitidi (il default è già 1.0; 2.0 dimezza la memoria su aree grandi) | sempre, per qualità |
| `--palette-photos DIR` | estrae dalle tue foto della zona i cluster cromatici di tetti e muri — i muri ricevono i loro toni intonaco, i tetti i colori veri delle tegole | bastano 3-10 foto; ideale un mix aereo + livello strada |
| `--reconstruct-buildings` | estrae footprint aggiuntivi dalle immagini satellitari dove OSM/Overture non hanno nulla | aree scarse/non mappate |
| `--imagery eox` | Sentinel-2 cloudless come sorgente immagini (gratuito, ~10 m/px → solo strutture grandi) | quando non hai di meglio |
| `--imagery "xyz:URL"` | tile XYZ ad alta risoluzione (~0.3-0.6 m/px) — i termini d'uso del provider sono responsabilità tua | ricostruzione seria |
| `--ortho file.tif --ortho-utc …` | tua ortofoto georeferenziata **con data/ora di acquisizione** → altezze dalle ombre per gli edifici non taggati + palette tetti + spot verdi vegetazione | il singolo upgrade migliore se hai ortofoto regionali |
| *(automatico)* `--no-ortho-normalize` per disattivare | le strisciate dei mosaici compositi (esposizione/tono diversi per acquisizione) vengono rilevate e allineate radiometricamente prima di ogni uso — lo stesso tetto è riconosciuto sia nella strisciata chiara che in quella scura | lascialo attivo; disattiva solo per immagini a acquisizione singola di cui ti fidi |
| *(automatico)* | gli edifici senza altezza ereditano la mediana dei vicini taggati entro ~250 m; i POI stampano archetipi landmark (cupole, guglie, merlature); le classi verde+foresta OSM/Overture diventano chioma 3D | — |

Il build termina con un riepilogo statistiche (edifici, altezze per
sorgente, footprint ricostruiti, vegetazione, POI, pixel della palette).
Dettaglio completo in `artifacts/gis/budapest_hq/meta.json`.

### A3. Controllo visivo prima dell'export

Renderizza qualche vista aerea direttamente dallo store costruito (non
serve training) — adattato da `scripts/test_budapest.py`:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
import numpy as np, cv2
from dronecv.gis.world import GisWorld
from dronecv.sim.headless import rasterizer
w = GisWorld.open("artifacts/gis/budapest_hq")
rgb, _ = rasterizer.render(w, np.array([0.0, 150.0, 0.0]), 45.0, 25.0,
                           960, 720, 70.0, 160.0, 55.0)
cv2.imwrite("check.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
EOF
```

Verifica: tetti terracotta vs muri chiari, acqua scura e piatta, verde dove
ci sono i parchi, sagome dei landmark dove te li aspetti.

### A4. Export

```bash
# asset Unity Terrain + scene.glb (glTF 2.0, materiali PBR, texture facciate)
dronecv gis export-scene --env budapest_hq

# .blend nativo — lancia Blender headless se è nel PATH,
# altrimenti stampa il comando esatto da eseguire dove c'è
dronecv gis export-blender --env budapest_hq --blend budapest.blend
```

La cartella di export (`artifacts/gis/budapest_hq/scene_export/`) contiene:

- `scene.glb` — terreno (vertex color per classe) + edifici LoD2 raggruppati
  per classe, materiali PBR dalla palette, texture procedurale finestre
  sulle facciate. Si apre direttamente in Blender, qualunque viewer glTF,
  three.js.
- `terrain.raw`, `splatmap.png`, `trees.json` — asset Unity Terrain.
- `buildings.obj` — fallback OBJ puro (senza materiali), amichevole per CAD.
- `blender_build_scene.py`, `scene_meta.json` (anchor, estensioni, posizione
  solare reale, attribuzioni).

**Unity**: apri il progetto → installa glTFast → menu **DroneCV > Import
GIS Scene…** → seleziona la cartella di export. Ottieni il Terrain con
orografia + layer splat + alberi tipizzati, e la scena glb con i materiali.

**Blender**: `export-blender` ha già prodotto il `.blend` (edifici LoD2 con
materiali, foreste istanziate, sole alla posizione solare reale, cielo).
Alternativa manuale:
`blender --background --python scene_export/blender_build_scene.py -- scene_export out.blend`.

### A5. Uso commerciale — da leggere

La licenza degli output è ereditata dalle sorgenti dati (dettagli in
[3d_product.md](3d_product.md)): i dati OSM/Overture sono ODbL (conserva le
stringhe di attribuzione da `scene_meta.json`), il DEM Copernicus consente
l'uso commerciale con credito, **le immagini EOX Sentinel-2 sono
non-commerciali** — per un prodotto vendibile alimenta palette e
ricostruzione con immagini tue o con licenza.

---

## Parte B — Test manuale di riconoscimento "da velivolo", end to end

Obiettivo: dimostrare che il localizzatore trasforma ciò che vede la camera
del drone in coordinate WGS84 + heading + confidenza onesta, senza GPS e
senza che nessuno gli dica mai dov'è.

### B1. Costruire l'ambiente

Stesso build della Parte A (A2). Da rete normale il percorso OSM porta in
un colpo solo anche POI e landcover:

```bash
dronecv gis build --bbox 47.4925,19.0290,47.5105,19.0560 --env-name budapest_test --res 1.0
```

### B2. Training

```bash
# tutto in un comando (capture -> training auto-dimensionato -> flight test -> report):
dronecv run-all --env budapest_test --budget 3000

# oppure passo per passo:
dronecv train --env budapest_test --budget 3000     # active loop, round auto-dimensionati
dronecv evaluate --env budapest_test                # metriche su probe freschi
```

Linee guida sul budget (capture, non epoche): **1200 = test fumo**
(aspettati centinaia di metri di errore), **3000+ = serio** per una città
densa di ~2×2 km; meno per aree con landmark forti (il prior di saliency
concentra già le capture dove servono). Per fedeltà maggiore alza
`sim.image_width/image_height` a 192 e `training.backbone_width` a 64 in
`configs/envs/budapest_test.yaml`.

### B3. Il test manuale con le foto

Prendi una foto in stile aereo dell'area (o usa i campioni committati in
`tests/data/budapest_photos/`) e chiedi le coordinate:

```bash
dronecv localize-photo tests/data/budapest_photos/03_parliament_aerial.jpg \
    --env budapest_test
```

Output: `lat, lon, alt (MSL), heading (nord vero), confidenza [0-1],
sigma (m)` e un link Google Maps. La versione completamente automatizzata
del test (build + 10 rendering casuali + training + tutte le foto +
confronti side-by-side + `results.json`):

```bash
python scripts/test_budapest.py --source osm --budget 3000 --res 1.0 \
    --photos tests/data/budapest_photos
```

**Domain gap**: i modelli addestrati su rendering sintetici vedono le foto
reali come un mondo diverso. Il filtro di preprocessing condiviso riduce il
divario ed è applicato in modo identico a training e inferenza (non può
divergere — la spec è registrata nel bundle del modello). Trova la
combinazione migliore per la tua area:

```bash
python scripts/filter_search.py --epochs 30
```

poi ri-addestra col vincitore, es. `dronecv train --env budapest_test`
dopo aver impostato nello YAML dell'ambiente:

```yaml
training:
  filter_mode: gray_edge   # none | gray | edge | gray_edge
  filter_edge_weight: 0.5
```

### B4. Modalità live "companion computer"

Terminale 1 — il mondo (sim headless, o una scena Unity che parla lo stesso
protocollo):

```bash
dronecv sim --env budapest_test
```

Terminale 2 — il localizzatore, che emette fix in streaming esattamente
come farebbe a bordo (frame camera in ingresso, coordinate in uscita; non
riceve mai una posizione):

```bash
dronecv localize --env budapest_test
```

Variante col telefono: `dronecv serve --env budapest_test` avvia il server
HTTP di inferenza; l'app Android (`android/`) fotografa, carica e mostra la
posizione stimata su Google Maps — oppure gira tutta on-device dopo
`dronecv export --env budapest_test` (bundle ONNX).

### B5. Il verdetto di affidabilità automatico

```bash
dronecv test-flight --env budapest_test --episodes 12
```

L'harness vola episodi scriptati + autonomi; la ground truth va SOLO
all'harness di test (il localizzatore è cieco per costruzione — il
protocollo rifiuta il canale truth ai ruoli non-harness). Il report
(`artifacts/budapest_test/report/`) fornisce percentili di errore
orizzontale/verticale, errore di heading, calibrazione della confidenza
(ECE), successo nel raggiungimento dei target, e un verdetto PASS/FAIL
sulle soglie dell'ambiente.

### B6. Leggere i numeri onestamente

- **La confidenza è calibrata**: 0.9 significa "il 90% dei fix con questa
  confidenza cade entro il raggio target". Su foto reali con budget di
  training piccolo aspettati confidenza BASSA — è il sistema che dice la
  verità, non un bug.
- **sigma_h_m** ha un pavimento pari all'errore di generalizzazione
  misurato del bundle: non può dichiarare una precisione mai dimostrata
  sui probe.
- Fidati di un fix quando: confidenza alta E retrieval/APR concordano
  (`disagreement_m` piccolo nei diagnostici) E la vista renderizzata dalla
  posa stimata somiglia alla foto.

---

## Risoluzione problemi

| Sintomo | Rimedio |
|---|---|
| Overpass in timeout / 403 (proxy aziendale o cloud) | usa i provider Overture: `scripts/test_budapest.py --source overture`; per `gis build` Overture viene selezionato automaticamente quando Overpass è irraggiungibile nei check di copertura — oppure builda da rete normale |
| `--imagery eox` non si connette | la rete blocca gli host fuori allowlist; esegui da rete normale o fornisci `--ortho` |
| Training lento su CPU | abbassa `--budget`, tieni `image_width` a 128; oppure usa una macchina CUDA (nessuna modifica al codice) |
| `export-blender` dice che Blender non c'è | installa Blender e rilancia, o copia il comando `blender --background …` stampato su una macchina che lo ha |
| Crash pyarrow nella scansione Overture | già mitigato (worker in processi isolati anti-crash); se persiste rilancia — le scansioni parziali riprovano file per file |
| Unity importa il glb senza materiali | installa `com.unity.cloud.gltfast` prima dell'import |
