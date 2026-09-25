"""
main.py -- CLI, wiring del processo candidato+render (punto 7). Un solo eccitatore +
una sola forma di risonatore attivi per l'intera sessione, scelti manualmente da riga
di comando (nessun routing IA sulla selezione, vedi agents.AgentManager -- vive solo
nel processo figlio, candidate_process.py).

2026-09-16 (diagnosi crackle, vedi test_audio_pipeline.py e candidate_process.py):
candidato+SPSA+render girano in un processo separato dallo stream audio, non piu' in
un thread in-process -- elimina la contesa sul GIL tra il render Python (fino a >1s
per alcuni eccitatori) e il callback sounddevice."""
import argparse
import sys

from agents import DESCRIPTOR_KEYS, EXCITER_PARAM_RANGES, RESONATOR_SHAPES, descriptor_keys_for
from audio_input import AudioDescriptorSource
from candidate_process import CandidateProcessHandle
from descriptor_input import DescriptorInput
from param_candidate import AUTO_TRIGGER_MIN_HOLD, AUTO_TRIGGER_THRESHOLD
from play_engine import LIMITER_CEILING, PlayEngine
from tuning import BUILTIN_SCALES, make_quantizer


def _relevant_keys(exciter_name, resonator_shape):
    """Unione delle chiavi rilevanti per l'eccitatore e per la forma di risonatore
    attivi (punto 1: il set dipende dall'agente attivo, via descriptor_keys_for)."""
    return sorted(set(descriptor_keys_for(exciter_name)) |
                  set(descriptor_keys_for(f"resonator_{resonator_shape}")))


def main():
    p = argparse.ArgumentParser(
        description="Prototipo motore realtime multiPhiMo -- valida l'architettura "
                    "(thread disaccoppiati, routing per descrittore, handoff candidato), "
                    "non il porting nativo (fase successiva).")
    p.add_argument("--exciter", default=None, choices=sorted(EXCITER_PARAM_RANGES),
                    help="fisso per la sessione (default: nessuno -- vedi --resonator per la modalita' auto)")
    p.add_argument("--resonator", default=None, choices=sorted(RESONATOR_SHAPES),
                    help="fisso per la sessione (default: nessuno)")
    p.add_argument("--selector-csv", default="selector_dataset.csv",
                    help="dataset del selettore (gen_selector_dataset.py), usato solo in modalita' auto")
    p.add_argument("--selector-k", type=int, default=30)
    p.add_argument("--selector-bias-alpha", type=float, default=0.3,
                    help="correzione anti-sbilanciamento per eccitatore ""(0=nessuna, vedi pair_selector.PairSelectorKnn)")
    p.add_argument("--auto-pair-min-hold", type=float, default=2.0,
                    help="secondi minimi tra due switch di coppia auto-selezionata")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--osc-ip", default="0.0.0.0")
    p.add_argument("--osc-port", type=int, default=9000)
    p.add_argument("--osc-address", default="/multiphimo/target")
    p.add_argument("--target-json", default=None,
                    help="target di test (JSON, {descrittore: valore}) per lavorare senza client OSC")
    p.add_argument("--spsa-iterations", type=int, default=4)
    p.add_argument("--refine-period", type=float, default=0.3,
                    help="secondi tra un ciclo di raffinamento e l'altro (0.2-0.5 consigliato)")
    p.add_argument("--max-voices", type=int, default=10)
    p.add_argument("--fade-ms", type=float, default=8.0, help="fade in/out (ms) sul render per evitare click")
    p.add_argument("--gain", type=float, default=0.4,
                    help="attenuazione fissa sul mix finale prima del clip di sicurezza (evita clipping con piu' voci in overlap)")
    p.add_argument("--limiter-ceiling", type=float, default=LIMITER_CEILING,
                    help="picco massimo assoluto dopo il gain (limiter, indipendente dal gain scelto)")
    p.add_argument("--latency", default="high",
                    help="latency di sounddevice.OutputStream ('low'/'high' o secondi): piu' alta = piu' margine "
                         "contro click/underrun causati dal thread SPSA in background, a costo di un filo di latenza in piu'")
    p.add_argument("--auto-trigger", action="store_true",
                    help="suona automaticamente ad ogni cambio sostanziale del candidato "
                         "(approssima 'seguire la curva', priorita' 3 punto 2) -- si aggiunge a Invio, non lo sostituisce")
    p.add_argument("--auto-trigger-threshold", type=float, default=AUTO_TRIGGER_THRESHOLD,
                    help="soglia di distanza normalizzata (0-1) oltre la quale un cambio e' 'sostanziale'")
    p.add_argument("--auto-trigger-min-hold", type=float, default=AUTO_TRIGGER_MIN_HOLD,
                    help="secondi minimi tra due auto-trigger (limita il rumore di SPSA)")
    p.add_argument("--morph", action="store_true",
                    help="crossfade tra una nota e la successiva invece di farle suonare in overlap indipendente "
                         "(consigliato insieme a --auto-trigger, evita che i suoni si sovrappongano stonati)")
    p.add_argument("--morph-ms", type=float, default=150.0,
                    help="durata (ms) del crossfade con --morph")
    p.add_argument("--audio-in", action="store_true",
                    help="acquisisce i descrittori target dal microfono invece di OSC/--target-json "
                         "(vedi audio_input.py) -- puo' coesistere con OSC, non esclusivo")
    p.add_argument("--smoothing-ms", type=float, default=200.0,
                    help="costante di tempo (ms) del fade tra un set di descrittori audio-in "
                         "e il successivo, usato solo con --audio-in")
    p.add_argument("--scale", default=None, choices=sorted(BUILTIN_SCALES),
                    help="se impostato, arrotonda il pitch target alla nota piu' vicina di "
                         "questa scala (fisso per la sessione, vedi tuning.py -- per il toggle "
                         "live usa gui.py)")
    p.add_argument("--a4", type=float, default=440.0,
                    help="riferimento 'La' (Hz) per --scale, es. 440/442")
    args = p.parse_args()

    auto_pair = args.exciter is None or args.resonator is None
    if auto_pair:
        # nessuna selezione manuale: il processo figlio sceglie la coppia
        # (pair_selector.py) non appena arriva il primo target -- fino ad allora
        # niente candidato (play_engine gestisce gia' candidate is None). Servono
        # tutti e 13 i descrittori in ingresso (il selettore, non un singolo agente,
        # decide quali contano), non solo quelli di una coppia non ancora scelta.
        keys = list(DESCRIPTOR_KEYS)
    else:
        keys = _relevant_keys(args.exciter, args.resonator)
    descriptor_input = DescriptorInput(keys, ip=args.osc_ip, port=args.osc_port, address=args.osc_address)
    if args.scale:
        descriptor_input.quantize_pitch = make_quantizer(args.scale, a4=args.a4)
        print(f"[main] Scale ON: pitch quantizzato su {args.scale} (La={args.a4}Hz)", file=sys.stderr)
    if args.target_json:
        descriptor_input.load_json(args.target_json)
        print(f"[main] target di test caricato da {args.target_json}", file=sys.stderr)
    descriptor_input.start_osc()  # se python-osc manca, stampa un avviso e prosegue solo col JSON

    audio_source = None
    if args.audio_in:
        audio_source = AudioDescriptorSource(descriptor_input, smoothing_ms=args.smoothing_ms)
        audio_source.start()
        print("[main] audio-in ON: target dal microfono (vedi audio_input.py)", file=sys.stderr)

    candidate_proc = CandidateProcessHandle(
        descriptor_input, weights_dir=args.weights_dir, dataset_dir=args.dataset_dir,
        initial_exciter=args.exciter, initial_resonator=args.resonator,
        spsa_iterations=args.spsa_iterations, refine_period=args.refine_period,
        fade_ms=args.fade_ms, auto_pair=auto_pair, selector_csv=args.selector_csv,
        selector_k=args.selector_k, selector_bias_alpha=args.selector_bias_alpha,
        auto_pair_min_hold=args.auto_pair_min_hold,
        auto_trigger=args.auto_trigger, auto_trigger_threshold=args.auto_trigger_threshold,
        auto_trigger_min_hold=args.auto_trigger_min_hold,
        # 2026-09-18: --morph/--morph-ms prima pilotavano solo PlayEngine (percorso
        # sync legacy, morto per questo main.py che usa candidate_proc per il render
        # vero) -- ora attivano il morphing B/A intra-coppia nel processo figlio, vedi
        # claude/morphing_spettrale_ricerca_proposta.md.
        morph=args.morph, morph_ms=args.morph_ms)
    engine = PlayEngine(candidate_proc, max_voices=args.max_voices, fade_ms=args.fade_ms,
                         morph=args.morph, morph_ms=args.morph_ms, gain=args.gain,
                         latency=args.latency, limiter_ceiling=args.limiter_ceiling)
    engine.start()
    # render+trigger ora nel processo figlio: il buffer gia' renderizzato arriva qui
    # via audio_sink, mai un render sincrono nel processo che ha anche lo stream audio.
    candidate_proc.audio_sink = engine.push_audio
    candidate_proc.start()

    if auto_pair:
        print(f"[main] modalita' auto: coppia scelta dal selettore ({args.selector_csv})", file=sys.stderr)
    else:
        print(f"[main] eccitatore={args.exciter}  risonatore={args.resonator}", file=sys.stderr)
    print(f"[main] target OSC atteso su {args.osc_ip}:{args.osc_port}{args.osc_address}/<descrittore> "
          f"(un float per messaggio)", file=sys.stderr)
    if args.auto_trigger:
        print(f"[main] auto-trigger ON (soglia={args.auto_trigger_threshold}): suona da solo "
              f"ad ogni cambio sostanziale del candidato", file=sys.stderr)
    print("[main] premi Invio per suonare una nota, Ctrl+C o Ctrl+D per uscire", file=sys.stderr)

    try:
        while True:
            line = sys.stdin.readline()
            if line == "":  # EOF (Ctrl+D)
                break
            candidate_proc.request_trigger()
    except KeyboardInterrupt:
        pass
    finally:
        print("[main] arresto...", file=sys.stderr)
        descriptor_input.stop()
        if audio_source is not None:
            audio_source.stop()
        candidate_proc.stop()
        engine.stop()


if __name__ == "__main__":
    main()
