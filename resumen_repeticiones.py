"""
Resume las repeticiones de E5a: lee las lineas RESULTADO_FINAL de
logs/e5a_seed*.log y calcula media y desviacion estandar muestral.

Uso (en el HPC, dentro de la carpeta del proyecto):
    python3 resumen_repeticiones.py
    python3 resumen_repeticiones.py --incluir-original   # suma la corrida original de E5a (87.16%)
"""

import argparse
import glob
import re
import statistics

PATRON = re.compile(
    r"RESULTADO_FINAL split_seed=(\S+) train_seed=(\S+) "
    r"test_acc=([\d.]+) test_f1_macro=([\d.]+) test_f1_weighted=([\d.]+)"
)


def leer_corridas(patron_logs):
    corridas = []
    for ruta in sorted(glob.glob(patron_logs)):
        with open(ruta, encoding="utf-8", errors="ignore") as f:
            lineas = [m for m in (PATRON.search(l) for l in f) if m]
        if not lineas:
            print(f"[aviso] {ruta}: sin linea RESULTADO_FINAL (corrida incompleta o fallida)")
            continue
        m = lineas[-1]
        corridas.append({
            "origen": ruta,
            "train_seed": m.group(2),
            "acc": float(m.group(3)),
            "f1_macro": float(m.group(4)),
            "f1_weighted": float(m.group(5)),
        })
    return corridas


def media_std(valores):
    if len(valores) < 2:
        return (valores[0] if valores else float("nan")), float("nan")
    return statistics.mean(valores), statistics.stdev(valores)  # desviacion muestral (n-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs/e5a_seed*.log")
    p.add_argument("--incluir-original", action="store_true",
                   help="Incluye la corrida original de E5a (accuracy 0.8716, F1 macro 0.8173; "
                        "sin semilla de entrenamiento fijada)")
    args = p.parse_args()

    corridas = leer_corridas(args.logs)
    if args.incluir_original:
        corridas.append({"origen": "E5a original", "train_seed": "sin fijar",
                         "acc": 0.8716, "f1_macro": 0.8173, "f1_weighted": None})

    if not corridas:
        print("No se encontraron corridas.")
        return

    print(f"{'Corrida':<16}{'Accuracy':>10}{'F1 macro':>10}{'F1 pond.':>10}")
    for c in corridas:
        etiqueta = f"seed {c['train_seed']}" if c["f1_weighted"] is not None else "E5a original"
        fw = f"{c['f1_weighted']:.4f}" if c["f1_weighted"] is not None else "   -"
        print(f"{etiqueta:<16}{c['acc']:>10.4f}{c['f1_macro']:>10.4f}{fw:>10}")

    acc_m, acc_s = media_std([c["acc"] for c in corridas])
    f1_m, f1_s = media_std([c["f1_macro"] for c in corridas])
    fw_vals = [c["f1_weighted"] for c in corridas if c["f1_weighted"] is not None]
    fw_m, fw_s = media_std(fw_vals)

    print(f"\nn = {len(corridas)} corridas (desviacion estandar muestral)")
    print(f"Accuracy : {acc_m * 100:.2f} % ± {acc_s * 100:.2f}")
    print(f"F1 macro : {f1_m:.4f} ± {f1_s:.4f}")
    print(f"F1 pond. : {fw_m:.4f} ± {fw_s:.4f}  (n = {len(fw_vals)})")


if __name__ == "__main__":
    main()
