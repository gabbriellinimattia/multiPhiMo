# MultiPhiMo (VST3)

Versione 0.1.0 · Licenza ISC · © 2026 Mattia Gabbriellini

Sintetizzatore a modelli fisici (10 eccitatori × 7 risonatori) pilotato da agenti IA:
si impostano valori di **descrittori spettrali** (centroide, rugosità, pitch, formanti, …)
e gli agenti scelgono i parametri fisici che li producono. Include una modalità Manual
(controllo diretto dei parametri fisici), scale/intonazioni (anche `.scl`), analisi
dell'audio in ingresso (sidechain), morph spettrale continuo, scelta automatica della
coppia eccitatore/risonatore, MIDI CC e OSC.

Formato: VST3 strumento (VST3i). Il bundle contiene già tutti i dati necessari
(pesi dei modelli e corpus in `Contents/Resources/data`).

---

## macOS (Apple Silicon: M1 e successivi, macOS 11 Big Sur o più recente)

1. Scompatta `MultiPhiMo-macOS-arm64.zip`.
2. Copia `MultiPhiMo.vst3` in `~/Library/Audio/Plug-Ins/VST3/`
   (Finder → Vai → Vai alla cartella… → incolla il percorso).
3. **Sblocca il plugin (una sola volta).** Il plugin è open source e non è firmato con un
   certificato Apple a pagamento, quindi macOS lo mette in quarantena quando lo scarichi
   e la DAW non riesce a caricarlo. Apri il Terminale ed esegui:

   ```
   xattr -dr com.apple.quarantine ~/Library/Audio/Plug-Ins/VST3/MultiPhiMo.vst3
   ```

4. Nella DAW rifai la scansione dei plugin (in Reaper: Preferenze → Plug-ins → VST →
   *Clear cache/re-scan*).

Al primo avvio macOS può chiedere se consentire le connessioni in entrata: servono solo
all'OSC (porta UDP 9000). Puoi rifiutare, il plugin funziona lo stesso.

I Mac Intel non sono ancora supportati (build solo arm64).

### Compilare da sorgente (macOS)

Servono gli strumenti da riga di comando di Xcode (`xcode-select --install`).

```
cd native/plugins/MultiPhiMo
make vst3            # build di sviluppo -> native/bin/MultiPhiMo.vst3
make release         # build pulita per macOS 11+, firma ad hoc, zip
```

`make release` produce `native/bin/MultiPhiMo-<versione>-macOS-arm64.zip` con il plugin,
questo README e le licenze. La versione si cambia in un solo punto: `VERSION_MAJOR/MINOR/PATCH`
nel `Makefile` del plugin.

---

## Linux (x86_64) — solo da sorgente, non ancora testato

Non esiste ancora un pacchetto precompilato. Il codice usa solo componenti portabili
(DPF, oscpack in variante POSIX), ma la build su Linux non è stata ancora provata:
segnala eventuali errori.

1. Dipendenze (Debian/Ubuntu):

   ```
   sudo apt-get install build-essential pkg-config git \
        libgl1-mesa-dev libx11-dev libxext-dev libxrandr-dev libxcursor-dev libdbus-1-dev
   ```

2. Compilazione:

   ```
      cd native/plugins/MultiPhiMo
   make vst3
   ```

3. Installazione: copia `native/bin/MultiPhiMo.vst3` in `~/.vst3/`
   (oppure `/usr/lib/vst3/` per tutti gli utenti) e rifai la scansione nella DAW.

Su Linux non c'è quarantena: non serve nessuno sblocco.

---

## Windows (x64) — non ancora supportato

La build per Windows non è ancora pronta: il Makefile compila la parte di rete OSC nella
variante POSIX (`oscpack/ip/posix`); per Windows vanno usati i file `oscpack/ip/win32` e
le librerie `ws2_32`/`winmm`. È nella lista dei prossimi lavori.

Quando sarà disponibile:

1. Copia la cartella `MultiPhiMo.vst3` in `C:\Program Files\Common Files\VST3\`.
2. **Sblocca i file scaricati (una sola volta).** Windows marca i file presi da Internet e
   può impedirne il caricamento. In PowerShell:

   ```
   Get-ChildItem -Recurse "C:\Program Files\Common Files\VST3\MultiPhiMo.vst3" | Unblock-File
   ```

   (in alternativa: tasto destro sullo zip scaricato → Proprietà → *Annulla blocco*,
   prima di scompattarlo).
3. Rifai la scansione dei plugin nella DAW. Se SmartScreen avvisa che l'editore è
   sconosciuto, il motivo è lo stesso di macOS: il plugin non è firmato con un
   certificato a pagamento.

---

## Uso rapido

- **Mode Agent**: imposti i 15 descrittori, gli agenti scelgono i parametri fisici.
  **Mode Manual**: controlli direttamente i parametri dell'eccitatore e del risonatore.
- **Play** suona una nota con il pitch impostato; le note MIDI cambiano l'altezza solo con
  **Scale** attivo (scale incluse: edo12/24/31, perfect, harmonic, oppure un file `.scl`).
- **Audio In** (sidechain): i descrittori seguono l'audio in ingresso.
  Esempio in Reaper: traccia del plugin a 4 canali, ingresso MIDI, armata, monitoring On,
  sidechain sui canali 3-4 (Pin connector); traccia del microfono con un send ai canali
  3-4 della traccia del plugin e master send spento.
- **Morph**: tenendo premuta una nota, il suono muta in modo continuo seguendo i
  descrittori (velocità = Smoothing). **Auto pair**: sceglie da sé eccitatore e risonatore.
- Il motore lavora internamente a 44.1 kHz e converte al sample rate del progetto.

### MIDI CC

| CC | Parametro | CC | Parametro |
|---|---|---|---|
| 7 | Gain | 20-28 | parametri eccitatore 1-9 (Manual) |
| 9 | Auto pair (≥ 64 = on) | 80-83, 85-90 | parametri risonatore 1-10 (Manual) |
| 16 | Mode | 102-116 | 15 descrittori |
| 17 | Eccitatore | 117 | Audio In (≥ 64 = on) |
| 18 | Risonatore | 118 | Smoothing (20-1000 ms) |
| | | 119 | Morph (≥ 64 = on) |

### OSC

Porta UDP 9000, un messaggio per descrittore: `/multiphimo/target/<descrittore> <float>`.

---

## Licenza

MultiPhiMo è distribuito con licenza ISC (vedi `LICENSE`). Include software di terze parti
(DPF, pugl, NanoVG, stb, font DejaVu Sans, oscpack, PocketFFT) con licenze permissive,
riportate in `THIRD_PARTY_LICENSES.md`.
