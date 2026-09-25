import argparse
import csv
import json
import sys
from pathlib import Path

from .descriptors import analyze_file, analyze_file_windowed

FIELDS = [
    "file", "spectral_centroid", "spectral_spread", "spectral_rolloff",
    "spectral_flatness", "roughness", "harmonic_tension", "inharmonicity",
    "formant_f1", "formant_f2", "formant_f3",
    "mod_rate", "mod_depth", "attack_time", "decay_time", "decay_capped", "pitch",
]


def main():
    p = argparse.ArgumentParser(
        description="Analizzatore descrittori (sez.1 pipeline_multiPhiMo.txt)."
    )
    p.add_argument("input", help="file .wav singolo, oppure cartella se --batch")
    p.add_argument("--batch", action="store_true",
                    help="tratta input come cartella, processa tutti i .wav")
    p.add_argument("--out", help="CSV (batch) o JSON (file singolo) di output; default stdout")
    p.add_argument("--no-pitch", action="store_true", help="salta la stima del pitch")
    p.add_argument("--normalize", action="store_true", help="peak-normalize prima dell'analisi")
    p.add_argument("--windowed", action="store_true",
                    help="curva nel tempo (analyze_signal_windowed) invece di un valore singolo; solo file singolo, non --batch")
    p.add_argument("--win-s", type=float, default=0.5, help="durata finestra in s (con --windowed)")
    p.add_argument("--hop-s", type=float, default=0.3, help="passo tra finestre in s (con --windowed)")
    args = p.parse_args()

    if args.windowed and args.batch:
        print("--windowed non supportato insieme a --batch", file=sys.stderr)
        sys.exit(1)

    kwargs = dict(extract_pitch=not args.no_pitch, normalize=args.normalize)
    fieldnames = [f for f in FIELDS if f != "pitch" or not args.no_pitch]

    if args.batch:
        folder = Path(args.input)
        files = sorted(folder.glob("*.wav"))
        rows = []
        for f in files:
            try:
                d = analyze_file(str(f), **kwargs)
                d["file"] = f.name
                rows.append(d)
                print(f"OK  {f.name}", file=sys.stderr)
            except Exception as e:
                print(f"ERR {f.name}: {e}", file=sys.stderr)
        f_out = open(args.out, "w", newline="") if args.out else sys.stdout
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        if args.out:
            f_out.close()
        print(f"totale: {len(rows)}/{len(files)} render analizzati", file=sys.stderr)
    elif args.windowed:
        curve = analyze_file_windowed(args.input, win_s=args.win_s, hop_s=args.hop_s, **kwargs)
        text = json.dumps(curve, indent=2)
        if args.out:
            Path(args.out).write_text(text)
            print(f"scritto {args.out} ({len(curve)} finestre)", file=sys.stderr)
        else:
            print(text)
            print(f"# {len(curve)} finestre", file=sys.stderr)
    else:
        d = analyze_file(args.input, **kwargs)
        d["file"] = Path(args.input).name
        text = json.dumps(d, indent=2)
        if args.out:
            Path(args.out).write_text(text)
            print(f"scritto {args.out}", file=sys.stderr)
        else:
            print(text)


if __name__ == "__main__":
    main()
