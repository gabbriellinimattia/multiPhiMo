"""
descriptor_input.py -- ricezione dei descrittori target (punto 1 del prompt). I valori
arrivano COSTANTI (non audio): un dict {nome_descrittore: float} via OSC (python-osc),
oppure da un JSON di test via CLI per lavorare senza client OSC.

Esposizione lock-free: l'ultimo target ricevuto e' un oggetto immutabile
(DescriptorTarget) sostituito con una singola assegnazione di riferimento
(`self.latest = ...`) -- atomica in CPython grazie al GIL, nessun lock necessario per la
lettura da un altro thread (vedi runtime_architettura_realtime.md: pattern lock-free
standard in audio realtime, thread di background disaccoppiato dal thread "play").

Formato OSC scelto: un messaggio PER DESCRITTORE, indirizzo "{address}/{nome}" con un
solo argomento float (es. "/multiphimo/target/spectral_centroid" 1200.0). Piu' semplice
da generare/debuggare di un unico messaggio con argomenti (nome,valore) alternati; via
pattern-matching del dispatcher (python-osc) un solo handler copre tutti i descrittori.
Ogni messaggio fa un MERGE di una chiave sopra l'ultimo target noto (mai un reset totale):
chiavi non rilevanti per l'agente attivo (fuori da `relevant_keys`, cfr.
agents.descriptor_keys_for) sono ignorate, non fanno fallire l'update.
"""
import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class DescriptorTarget:
    """Immutabile: values e' una MappingProxyType (vista read-only), timestamp e'
    time.monotonic() al momento dell'update. Un nuovo target e' sempre un nuovo oggetto,
    mai una mutazione in-place."""
    values: MappingProxyType
    timestamp: float


class DescriptorInput:
    def __init__(self, relevant_keys, ip="0.0.0.0", port=9000, address="/multiphimo/target"):
        self.relevant_keys = set(relevant_keys)
        self.ip = ip
        self.port = port
        self.address = address.rstrip("/")
        self.latest = None  # DescriptorTarget o None finche' non arriva nulla
        self._server = None
        # quantizzazione pitch opzionale (2026-09-16, toggle "Scale" in gui.py): se
        # impostata (callable freq->freq, vedi tuning.py), applicata SOLO alla chiave
        # "pitch" in set_target, prima di salvarla in self.latest -- unico punto di
        # intercettazione che copre tutte e tre le sorgenti (slider manuali, OSC,
        # audio-in) senza duplicare logica, perche' tutte passano da qui.
        self.quantize_pitch = None

    # ---- update (usato sia dall'handler OSC sia da load_json) ----

    def set_target(self, raw: dict):
        """Merge di raw (filtrato su relevant_keys, valori non convertibili a float
        ignorati) sopra l'ultimo target noto. Assegnazione atomica del nuovo oggetto."""
        filtered = {}
        for k, v in raw.items():
            if k not in self.relevant_keys:
                continue
            try:
                filtered[k] = float(v)
            except (TypeError, ValueError):
                continue
        if "pitch" in filtered and self.quantize_pitch is not None:
            filtered["pitch"] = self.quantize_pitch(filtered["pitch"])
        if not filtered:
            return
        base = dict(self.latest.values) if self.latest is not None else {}
        base.update(filtered)
        self.latest = DescriptorTarget(values=MappingProxyType(base), timestamp=time.monotonic())

    def load_json(self, path):
        with open(path) as f:
            data = json.load(f)
        self.set_target(data)

    # ---- OSC ----

    def start_osc(self):
        """Avvia il server OSC in un thread separato (daemon). Ritorna False (con
        messaggio su stderr) se python-osc non e' installato -- in tal caso resta
        disponibile solo l'input da --target-json."""
        try:
            from pythonosc.dispatcher import Dispatcher
            from pythonosc.osc_server import ThreadingOSCUDPServer
        except ImportError:
            print("[descriptor_input] python-osc non installato (pip install python-osc): "
                  "input OSC disabilitato, uso solo --target-json.", file=sys.stderr)
            return False

        disp = Dispatcher()

        def _handler(osc_address, *args):
            if not args:
                return
            prefix = self.address + "/"
            if not osc_address.startswith(prefix):
                return
            name = osc_address[len(prefix):]
            self.set_target({name: args[0]})

        disp.map(f"{self.address}/*", _handler)
        self._server = ThreadingOSCUDPServer((self.ip, self.port), disp)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"[descriptor_input] OSC in ascolto su {self.ip}:{self.port}, "
              f"indirizzo {self.address}/<descrittore> (un float per messaggio)", file=sys.stderr)
        return True

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()  # rilascia subito la porta (serve per un restart pulito)
            self._server = None


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test manuale: stampa ogni update del target ricevuto via OSC.")
    p.add_argument("--ip", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--address", default="/multiphimo/target")
    p.add_argument("--keys", nargs="+", default=None, help="descrittori rilevanti (default: tutti i 13)")
    args = p.parse_args()

    from agents import DESCRIPTOR_KEYS

    di = DescriptorInput(args.keys or DESCRIPTOR_KEYS, args.ip, args.port, args.address)
    if not di.start_osc():
        sys.exit(1)
    print("In ascolto, Ctrl+C per uscire...", file=sys.stderr)
    try:
        last_ts = None
        while True:
            if di.latest is not None and di.latest.timestamp != last_ts:
                last_ts = di.latest.timestamp
                print(dict(di.latest.values))
            time.sleep(0.1)
    except KeyboardInterrupt:
        di.stop()
