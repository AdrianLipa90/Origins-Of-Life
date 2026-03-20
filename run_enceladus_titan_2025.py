"""
╔══════════════════════════════════════════════════════════════════════╗
║        TYTAN & ENCELADUS — ANALIZA MOŻLIWOŚCI POWSTANIA ŻYCIA        ║
║                    Dane: NASA/ESA 2024–2025                          ║
╚══════════════════════════════════════════════════════════════════════╝

Źródła danych:
  - Enceladus pH: Phosphates reveal high pH ocean water (GCA 2025)
  - Enceladus organics: Nature Astronomy 2025 (ice grain analysis)
  - Enceladus metals: JAMSTEC 2025 (Ni/Co/Cu depletion study)
  - Enceladus vents: >90°C hydrothermal activity (Cassini)
  - Tytan vesicles: Int. Journal of Astrobiology 2025 (Mayer & Nixon)
  - Tytan lakes: ESA — 1.5 atm, 94K (-179°C), 75% C2H6 + 10% CH4
  - Tytan methane crust: 2024 — do 10 km grubości
"""

import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    name: str
    code: str
    location: str
    temp_C: float
    pressure_atm: float
    UV_flux: float
    solvent: str
    pH: float
    redox: str
    energy_source: str
    k_energy: float
    catalyst: str
    k_catalysis: float
    concentration_boost: float
    k_synthesis: float
    k_degradation: float
    expected_protocells: int
    timescale_description: str


# ============================================================================
# ENCELADUS — zaktualizowane dane NASA 2025
# pH ocean: 8.5-9 (serpentynizacja, nowe badania GCA 2025)
# Temperatura: ~4°C ocean + wulkany hydrotermalne >90°C (Cassini)
# H2, CO2, NH3, CH4, HCN, złożone organiki (Nature Astronomy 2025)
# Metale: Ni/Co depleted (JAMSTEC 2025) — nieznacznie obniżamy k_catalysis
# ============================================================================
ENCELADUS = ScenarioConfig(
    name="Enceladus — Ocean Podlodowy (dane NASA 2025)",
    code="E",
    location="Enceladus (Saturn) — 40 km pod lodem, serpentyniacja",
    temp_C=4.0,            # bulk ocean; wulkany >90°C
    pressure_atm=800.0,    # głęboki ocean podlodowy
    UV_flux=0.0,
    solvent="H2O (ciekla woda, zasadowa)",
    pH=8.8,                # zaktualizowane: 8-9 (GCA 2025), poprzednio 9.5
    redox="Silnie redukujące (H2 z serpentynizacji)",
    energy_source="Hydrotermalne (pływowe podgrzewanie + serpentynizacja)",
    k_energy=0.10,         # zwiększone: H2 z serpentynizacji bardziej aktywny
    catalyst="Fe-S + Mg-krzemiany + HCN (Nature Astronomy 2025)",
    k_catalysis=10.5,      # nieznacznie obniżone z powodu deplecji Co/Cu (JAMSTEC)
    concentration_boost=650.0,
    k_synthesis=0.09,      # zwiększone: wykryto złożone organiki
    k_degradation=0.008,   # niska temp = wolna degradacja
    expected_protocells=480,
    timescale_description="Dni (bardzo sprzyjające warunki)"
)

# ============================================================================
# TYTAN — zaktualizowane dane ESA/NASA 2025
# Temperatura: 94K = -179°C, ciśnienie: 1.5 atm (ESA)
# Jeziora: ~75% C2H6, 10% CH4, 7% C3H8, HCN
# KLUCZOWE 2025: amfifyle mogą tworzyć pęcherzyki przez mgłę nad jeziorami
# (Mayer & Nixon, Int. J. Astrobiology 2025)
# ============================================================================
TYTAN = ScenarioConfig(
    name="Tytan — Jeziora Metanowe Kraken Mare (dane ESA/NASA 2025)",
    code="D",
    location="Tytan (Saturn) — Kraken Mare, -179°C, 1.5 atm",
    temp_C=-179.0,
    pressure_atm=1.5,      # zaktualizowane z ESA (poprzednio 1.45)
    UV_flux=2.5,           # UV dociera przez atmosferę, tworzenie tholinów
    solvent="CH4/C2H6 (ciekly metan/etan, 75%C2H6+10%CH4+HCN)",
    pH=7.0,                # pH nieaplikowalne w rozpuszczalniku nieaqueous
    redox="Nieutleniajace (brak O2)",
    energy_source="Fotochemia atmosferyczna + tholiny",
    k_energy=0.0015,       # zaktualizowane: nieco wyższe (UV + tholiny)
    catalyst="Tholiny + HCN + amfifyle (Int.J.Astrobiol. 2025)",
    k_catalysis=2.5,       # zaktualizowane: nowy mechanizm pęcherzykowania (2025)
    concentration_boost=120.0,  # zaktualizowane: koncentracja przez mgłę
    k_synthesis=0.0012,    # ekstremalnie wolne przy 94K
    k_degradation=0.00008, # ale i degradacja minimalna
    expected_protocells=45,  # zaktualizowane w górę (nowy mechanizm 2025)
    timescale_description="Tysiace godzin (bardzo wolno, ale mozliwe — 2025)"
)


# ============================================================================
# SYMULATOR (zaczerpnięty z OriginOfLife.v4.0_ET.py)
# ============================================================================
class UniversalOriginSimulator:
    def __init__(self, config: ScenarioConfig, Nx=64, Ny=64, dt_h=0.02):
        self.config = config
        self.Nx, self.Ny = Nx, Ny
        self.dt_h = dt_h
        self.t_h = 0.0
        self.E = self.O = self.N = self.R = self.M = self.L = self.Cat = None
        self.rna_population = []
        self.protocell_count = 0
        self.history = {
            'time_h': [], 'mean_R': [], 'mean_M': [],
            'n_polymers': [], 'n_protocells': [], 'mean_fitness': []
        }

    def initialize(self):
        temp_factor = np.exp((self.config.temp_C - 25) / 100.0)
        self.E = np.random.uniform(0.1, 0.3, (self.Nx, self.Ny)) * max(temp_factor, 0.01)
        self.O = np.random.uniform(0.05, 0.15, (self.Nx, self.Ny))
        self.N = np.random.uniform(0.01, 0.05, (self.Nx, self.Ny))
        self.R = np.zeros((self.Nx, self.Ny))
        self.M = np.zeros((self.Nx, self.Ny))
        self.L = np.random.uniform(0.005, 0.01, (self.Nx, self.Ny))
        self.Cat = np.random.uniform(0.8, 1.2, (self.Nx, self.Ny))
        n_seed = max(5, int(20 * max(temp_factor, 0.01)))
        for _ in range(n_seed):
            self.rna_population.append({
                'length': np.random.randint(20, 50),
                'position': (np.random.randint(0, self.Nx), np.random.randint(0, self.Ny)),
                'fitness': np.random.uniform(0.3, 0.6),
                'age': 0.0
            })

    def step_energy_conversion(self):
        k = self.config.k_energy
        efficiency = 0.8 if self.config.UV_flux > 0 else 0.6
        dE = -k * self.E * self.dt_h
        dO = k * self.E * efficiency * self.dt_h
        self.E = np.clip(self.E + dE, 0, 1)
        self.O = np.clip(self.O + dO, 0, 1)

    def step_catalysis(self):
        k_cat = self.config.k_catalysis * 0.1
        ce = self.Cat / (self.Cat + 1.0)
        dO = -k_cat * self.O * ce * self.dt_h
        dN = k_cat * self.O * ce * self.dt_h
        self.O = np.clip(self.O + dO, 0, 1)
        self.N = np.clip(self.N + dN, 0, 1)

    def step_polymerization(self):
        k_syn = self.config.k_synthesis
        boost = self.config.concentration_boost / 1000.0
        dN = -k_syn * self.N * boost * self.dt_h
        dR = k_syn * self.N * boost * self.dt_h
        self.N = np.clip(self.N + dN, 0, 1)
        self.R = np.clip(self.R + dR, 0, 1)

    def step_degradation(self):
        k_deg = self.config.k_degradation
        T_ref = 25.0
        temp_factor = np.exp(0.05 * (self.config.temp_C - T_ref))
        temp_factor = max(temp_factor, 1e-10)
        self.R = np.clip(self.R - k_deg * self.R * temp_factor * self.dt_h, 0, 1)
        for poly in list(self.rna_population):
            poly['age'] += self.dt_h
            if np.random.random() < k_deg * temp_factor * self.dt_h:
                self.rna_population.remove(poly)

    def step_membrane_formation(self):
        k_mem = 0.4
        T = self.config.temp_C
        if T < -100:
            # 2025: nowy mechanizm (Mayer & Nixon) - wyższy współczynnik dla Tytana
            temp_factor = 0.15  # wzrost z 0.1 -> 0.15
        elif T < 0:
            temp_factor = 0.5
        elif T < 70:
            temp_factor = 1.0
        elif T < 100:
            temp_factor = 0.7
        else:
            temp_factor = 0.3
        dL = -k_mem * self.L * temp_factor * self.dt_h
        dM = k_mem * self.L * temp_factor * self.dt_h
        self.L = np.clip(self.L + dL, 0, 1)
        self.M = np.clip(self.M + dM, 0, 1)

    def step_protocell_detection(self):
        pc = np.where((self.M > 0.05) & (self.R > 0.03))
        self.protocell_count = len(pc[0])

    def record_state(self):
        self.history['time_h'].append(self.t_h)
        self.history['mean_R'].append(np.mean(self.R))
        self.history['mean_M'].append(np.mean(self.M))
        self.history['n_polymers'].append(len(self.rna_population))
        self.history['n_protocells'].append(self.protocell_count)
        f = [p['fitness'] for p in self.rna_population]
        self.history['mean_fitness'].append(np.mean(f) if f else 0)

    def step(self):
        self.step_energy_conversion()
        self.step_catalysis()
        self.step_polymerization()
        self.step_degradation()
        self.step_membrane_formation()
        self.step_protocell_detection()
        self.t_h += self.dt_h

    def run(self, hours=120, record_interval=2.0):
        n_steps = int(hours / self.dt_h)
        record_steps = int(record_interval / self.dt_h)
        print(f"\n  Lokalizacja : {self.config.location}")
        print(f"  Temperatura : {self.config.temp_C}°C")
        print(f"  Rozpuszczalnik: {self.config.solvent}")
        print(f"  Symulacja   : {hours}h ({n_steps:,} kroków)\n")
        for step in range(n_steps):
            self.step()
            if step % record_steps == 0:
                self.record_state()
                if step % (record_steps * 10) == 0:
                    print(f"    t={self.t_h:7.1f}h | Polimery={len(self.rna_population):3d} | Proto-komórki={self.protocell_count:3d}")
        return pd.DataFrame(self.history)


# ============================================================================
# GŁÓWNA ANALIZA
# ============================================================================
def habitability_score(config, final_protocells):
    """Oblicz wskaźnik habitowalności 0-100."""
    score = 0
    # Woda ciekła lub alternatywny rozpuszczalnik
    if "H2O" in config.solvent:
        score += 25
    elif "CH4" in config.solvent or "C2H6" in config.solvent:
        score += 10
    # Zakres temperatury
    if -20 <= config.temp_C <= 120:
        score += 20
    elif -200 <= config.temp_C < -20:
        score += 5
    # Źródło energii
    if config.k_energy > 0.05:
        score += 20
    elif config.k_energy > 0.001:
        score += 10
    # Kataliza
    if config.k_catalysis > 8:
        score += 15
    elif config.k_catalysis > 2:
        score += 8
    # Proto-komórki
    ratio = final_protocells / max(config.expected_protocells, 1)
    score += min(20, int(20 * ratio))
    return min(score, 100)


if __name__ == "__main__":
    np.random.seed(42)  # Powtarzalność wyników

    print("=" * 70)
    print("   ANALIZA MOŻLIWOŚCI POWSTANIA ŻYCIA — TYTAN & ENCELADUS")
    print("   Parametry zaktualizowane wg danych NASA/ESA 2024–2025")
    print("=" * 70)

    scenarios = [ENCELADUS, TYTAN]
    results = {}
    summaries = []

    for cfg in scenarios:
        print(f"\n{'='*70}")
        print(f"  SCENARIUSZ {cfg.code}: {cfg.name}")
        print(f"{'='*70}")

        hours = 500 if cfg.code == "D" else 120
        sim = UniversalOriginSimulator(cfg, Nx=64, Ny=64, dt_h=0.02)
        sim.initialize()
        df = sim.run(hours=hours)
        results[cfg.code] = (sim, df)

        hab = habitability_score(cfg, sim.protocell_count)

        print(f"\n  WYNIKI KOCOWE:")
        print(f"    Polimery RNA-podobne : {len(sim.rna_population)}")
        print(f"    Proto-komórki        : {sim.protocell_count}")
        print(f"    Oczekiwano           : {cfg.expected_protocells}")
        pct = 100 * sim.protocell_count / max(cfg.expected_protocells, 1)
        print(f"    Skutecznosc          : {pct:.1f}%")
        print(f"    Wskaznik habitowalnosci: {hab}/100")

        summaries.append({
            'Cialo': 'Enceladus' if cfg.code == 'E' else 'Tytan',
            'Temp_C': cfg.temp_C,
            'Rozpuszczalnik': cfg.solvent[:30],
            'pH': cfg.pH,
            'Energia': cfg.energy_source[:30],
            'Polimery': len(sim.rna_population),
            'ProtoKomorki': sim.protocell_count,
            'Oczekiwano': cfg.expected_protocells,
            'Skutecznosc_%': round(pct, 1),
            'Habitowalnosc': hab,
            'Skala_czasu': cfg.timescale_description
        })

    # Raport końcowy
    print("\n" + "=" * 70)
    print("  RAPORT POROWNAWCZY — TYTAN vs ENCELADUS")
    print("=" * 70)

    df_summary = pd.DataFrame(summaries)
    print(df_summary.to_string(index=False))
    df_summary.to_csv("raport_tytan_enceladus_2025.csv", index=False)

    print("\n" + "=" * 70)
    print("  WNIOSKI (dane NASA/ESA 2024-2025)")
    print("=" * 70)

    e_sum = summaries[0]
    d_sum = summaries[1]

    print(f"""
ENCELADUS:
  + Ciekla woda (pH={ENCELADUS.pH}) — najwazniejszy warunek
  + Serpentynizacja dostarcza H2 — bezposrednie paliwo dla chemii
  + Wykryto HCN, złożone organiki (Nature Astronomy, marzec 2025)
  + Temperatura wulkanow >90°C — gradient energetyczny dla protobiochemii
  - Deplecja Co i Cu moze limitowac aktywnosc metanogenow (JAMSTEC 2025)
  - Brak swiatla UV (ocean pod 40 km lodu)
  WYNIK: {e_sum['ProtoKomorki']} proto-komorek | Habitowalnosc: {e_sum['Habitowalnosc']}/100
  OCENA: >>> BARDZO WYSOKA szansa na zycie <<<

TYTAN:
  + Bogata chemia organiczna (tholiny, HCN, acetylene)
  + NOWE 2025: amfifyle tworza pecherzyski przez mgłe nad jeziorami (Mayer & Nixon)
  + Atmosfera 1.5 atm — stabilna, bogata w N2/CH4
  + Misja Dragonfly (NASA, start 2028) bedzie testowac habitowalnosc in situ
  - Temperatura -179°C — reakcje chemiczne ekstremalnie wolne
  - Brak ciekłej wody na powierzchni
  - Biochemia bylaby RADYKALNIE inna od ziemskiej
  WYNIK: {d_sum['ProtoKomorki']} proto-komorek | Habitowalnosc: {d_sum['Habitowalnosc']}/100
  OCENA: >>> NISKA-UMIARKOWANA szansa, mozliwe 'zycie egzotyczne' <<<
""")

    print("=" * 70)
    winner = "ENCELADUS" if e_sum['Habitowalnosc'] >= d_sum['Habitowalnosc'] else "TYTAN"
    print(f"  ZWYCIEZCA ANALIZY: {winner}")
    print(f"  Zapisano: raport_tytan_enceladus_2025.csv")
    print("=" * 70)
