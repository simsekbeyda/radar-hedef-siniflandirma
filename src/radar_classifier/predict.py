"""Komut satırından hedef tahmini.

Kullanım:
    # Etkileşimli mod (değerleri tek tek sorar)
    python -m radar_classifier.predict

    # Argümanlarla
    python -m radar_classifier.predict --hiz 45 --irtifa 150 --rcs 0.03 --manevra 2.5 --pervane 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .config import ACIKLAMALAR, GUVEN_ESIGI
from .model import load_model, to_frame


@dataclass
class Tahmin:
    tur: str
    guven: float
    olasiliklar: dict[str, float]

    @property
    def ad(self) -> str:
        return ACIKLAMALAR[self.tur][0]

    @property
    def oneri(self) -> str:
        return ACIKLAMALAR[self.tur][1]

    @property
    def kontrol_gerekli(self) -> bool:
        return self.guven < GUVEN_ESIGI


def tahmin_et(model, hiz, irtifa, rcs, manevra, pervane) -> Tahmin:
    """Radar ölçümlerinden hedef türünü ve modelin güvenini döndürür."""
    X = to_frame(hiz, irtifa, rcs, manevra, pervane)
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    best = int(proba.argmax())
    return Tahmin(
        tur=classes[best],
        guven=float(proba[best] * 100),
        olasiliklar={c: float(p * 100) for c, p in zip(classes, proba)},
    )


def _sayi_al(soru: str, min_deger: float = 0.0) -> float:
    while True:
        try:
            deger = float(input(soru).replace(",", "."))
        except ValueError:
            print("  Uyarı: Lütfen bir sayı girin (örnek: 45 veya 0.03).")
            continue
        if deger < min_deger:
            print(f"  Uyarı: Değer {min_deger}'dan küçük olamaz.")
            continue
        return deger


def _yazdir(t: Tahmin) -> None:
    print("-" * 55)
    print(f"Tespit edilen hedef büyük olasılıkla: {t.ad}")
    print(f"   Modelin güveni: %{t.guven:.0f}")
    if t.kontrol_gerekli:
        print("   Uyarı: Güven düşük, sonucu bir operatör kontrol etmeli.")
    print(f"Öneri: {t.oneri}")
    print("   Olasılıklar: " + ", ".join(
        f"{ACIKLAMALAR[c][0]} %{p:.0f}" for c, p in
        sorted(t.olasiliklar.items(), key=lambda kv: -kv[1])
    ))
    print("=" * 55)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Radar hedef sınıflandırma aracı")
    p.add_argument("--hiz", type=float, help="Hız (km/saat)")
    p.add_argument("--irtifa", type=float, help="İrtifa (metre)")
    p.add_argument("--rcs", type=float, help="Radar kesit alanı (m²)")
    p.add_argument("--manevra", type=float, help="Manevra skoru (yön değişimi / dk)")
    p.add_argument("--pervane", type=int, choices=[0, 1], help="Pervane imzası (1=var, 0=yok)")
    args = p.parse_args(argv)

    model, _ = load_model()

    print("=" * 55)
    print("   RADAR HEDEF SINIFLANDIRMA ARACI")
    print("=" * 55)

    degerler = [args.hiz, args.irtifa, args.rcs, args.manevra, args.pervane]
    if all(v is not None for v in degerler):
        t = tahmin_et(model, *degerler)
    else:
        hiz = _sayi_al("Hız (km/saat)                     : ")
        irtifa = _sayi_al("İrtifa (metre)                    : ")
        rcs = _sayi_al("Radar kesit alanı (m²)            : ")
        manevra = _sayi_al("Manevra skoru (yön değişimi / dk) : ")
        cevap = input("Pervane imzası var mı? (e/h)      : ").strip().lower()
        t = tahmin_et(model, hiz, irtifa, rcs, manevra, 1 if cevap.startswith("e") else 0)

    _yazdir(t)


if __name__ == "__main__":
    main()
