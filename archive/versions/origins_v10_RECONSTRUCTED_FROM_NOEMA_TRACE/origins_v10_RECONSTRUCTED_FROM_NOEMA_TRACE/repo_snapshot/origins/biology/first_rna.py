"""
First RNA Emergence Model — predykcja i symulacja powstawania pierwszego replikatora.

Oparty na:
  - Joyce & Orgel (1999): minimalna długość do samoreplikacji ~40 nt
  - Ferris et al. (1996): montmorylonit katalizuje syntezę oligomerów do 50 nt
  - Higgs & Lehman (2015): model progowy — RNA world wymaga przekroczenia bariery
  - Szostak (2012): stochastyczna emergencja pierwszego replikatora

Mechanizm w tym modelu:
  1. OLIGOMERYZACJA: monomery łączą się stopniowo (Poisson process)
     k_ligation = f(temperatura, katalizator, topologia Blocha)
  2. PRÓG KRYTYCZNY: oligomer o długości >= L_min i GC >= gc_min
     staje się „zalążkiem replikatora" (proto-ribozyme)
  3. REPLIKACJA SZABLONOWA: pierwszy replikator produkuje kopie
     z wykładniczym wzrostem (logistic z pojemnością środowiska)
  4. HOLONOMIA: Berry phase akumulowana w TopologyField moduluje
     k_ligation — pole topologiczne przyspiesza lub hamuje łączenie

Predykcja: czas T_emergence do pierwszego replikatora jako funkcja
  warunków środowiskowych (temp, katalizator, topo_strength).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Zeta modulation + Heisenberg soft-clip (per Adrian: "tam gdzie zera tam
#    zeta, tam gdzie NaN soft clip heisenberga") ────────────────────────────────

_ZETA_EPSILON = 1e-12  # floor dla zerotych kinetyk

def _zeta_floor(x: float) -> float:
    """Nie pozwól stawce spaść do hard zero — zeta regularyzacja."""
    return max(x, _ZETA_EPSILON)


def _heisenberg_clip(x: float, sigma: float = 1e-9) -> float:
    """Soft-clip NaN/Inf → sygnał szumowy Heisenberga.

    Zamiast NaN w historii: mała ale niezerowa fluktuacja (zasada nieoznaczoności
    zapobiega egzaktnemu zeru energii kinetycznej).
    """
    if math.isnan(x) or math.isinf(x):
        import random
        return abs(random.gauss(0, sigma))
    return x


# ── Stałe biofizyczne ─────────────────────────────────────────────────────────

L_MIN_RIBOZYME  = 40    # minimalna długość do aktywności katalitycznej [nt]
L_MIN_REPLICATE = 35    # minimalna długość do replikacji szablonowej [nt]
GC_MIN_STABLE   = 0.35  # minimalne GC do stabilności struktury drugorzędowej
GC_OPT          = 0.50  # optymalne GC (maksymalna fitness)
K_LIG_BASE      = 3e-4  # bazowa stała ligacji [1/h per monomer]
K_HYDRO_BASE    = 0.08  # hydroliza oligomeru [1/h]
K_TEMPLATE_BASE = 0.05  # replikacja szablonowa [1/h]
CARRYING_CAPACITY = 500 # maks. liczba replikatorów w środowisku


# ── Struktury danych ──────────────────────────────────────────────────────────

@dataclass
class OligomerPool:
    """
    Rozkład długości oligomerów w środowisku prebiologicznym.

    Reprezentacja: tablica counts[i] = liczba oligomerów o długości i+1.
    Długości od 1 (monomer) do max_len.
    """
    max_len: int = 80
    counts: np.ndarray = field(default_factory=lambda: np.zeros(80, dtype=np.float64))
    monomer_pool: float = 1000.0  # pula wolnych monomerów

    @classmethod
    def seed(cls, monomer_conc: float = 1000.0, max_len: int = 80) -> "OligomerPool":
        pool = cls(max_len=max_len)
        pool.counts = np.zeros(max_len, dtype=np.float64)
        pool.counts[0] = monomer_conc * 0.1  # 10% zaczyna jako dimery
        pool.monomer_pool = monomer_conc * 0.9
        return pool

    def n_oligomers(self, min_len: int = 2) -> float:
        return float(self.counts[min_len-1:].sum())

    def mean_length(self) -> float:
        lengths = np.arange(1, self.max_len + 1, dtype=float)
        total = float(self.counts.sum())
        if total < 1e-9:
            return 0.0
        return float((self.counts * lengths).sum() / total)

    def n_above_threshold(self, L_min: int) -> float:
        return float(self.counts[L_min-1:].sum())


@dataclass
class EmergenceState:
    """Stan procesu emergencji pierwszego RNA."""
    t_h: float = 0.0
    oligomer_pool: OligomerPool = field(default_factory=OligomerPool.seed)
    n_replicators: float = 0.0     # liczba aktywnych replikatorów
    first_replicator_t: Optional[float] = None   # czas emergencji
    berry_at_emergence: Optional[float] = None   # holonomia w chwili emergencji
    n_ribozymes: float = 0.0       # oligomery >= L_MIN_RIBOZYME
    gc_mean: float = 0.42          # średnie GC w populacji oligomerów
    history: dict = field(default_factory=lambda: {
        'time_h': [], 'mean_length': [], 'n_above_Lmin': [],
        'n_replicators': [], 'berry': [], 'gc_mean': [],
        'k_lig_eff': [],
    })


# ── Kinetyka ligacji zmodulowana Blochem ──────────────────────────────────────

def k_ligation_effective(
    temp_C: float,
    k_catalysis: float,
    bloch_coherence: float,
    berry_accumulated: float,
    gc_mean: float,
    concentration_boost: float = 1000.0,
) -> float:
    """
    Efektywna stała ligacji monomeru do oligomeru.

    Składowe:
    - Temperatura: Arrhenius, E_a = 60 kJ/mol (Ferris 1996)
    - Katalizator + concentration_boost: lokalna konc. na powierzchni
      minerału do 1000x (Ferris 1996) — kluczowa bariera wzrostu
    - Bloch coherence: geometryczny operator fazy pola Kählera
    - Berry phase: akumulowana holonomia — rezonans topologiczny
    - GC bias: stabilność struktury drugorzędowej RNA
    """
    T_K = max(1.0, temp_C + 273.15)
    E_a = 60_000.0
    R_gas = 8.314
    T_ref = 338.15
    arrhenius = math.exp(-E_a / R_gas * (1.0/T_K - 1.0/T_ref))

    cat_factor = math.log1p(k_catalysis) / math.log1p(15.0)

    # Koncentracja lokalna na powierzchni minerału (Ferris 1996).
    # Na montmorylonicie lokalna konc. nukleotydów rośnie do 1000x.
    # To przesuwa kinetykę z reżimu hydrolizy (k_lig*M < k_hyd)
    # do reżimu wzrostu (k_lig*M*boost >> k_hyd*n_bonds).
    # Używamy sqrt(boost) bo realnie tylko frakcja oligomerów jest na powierzchni.
    conc_factor = math.sqrt(max(1.0, concentration_boost)) / math.sqrt(1000.0) * 10.0

    bloch_factor = 1.0 + 2.0 * bloch_coherence

    berry_factor = 1.0 + abs(berry_accumulated) * 0.8

    gc_factor = math.exp(-((gc_mean - GC_OPT)**2) / 0.05) + 0.3

    result = K_LIG_BASE * arrhenius * cat_factor * conc_factor * bloch_factor * berry_factor * gc_factor
    # Zeta: zerowe stawki zamieniamy na epsilon, nie hard-0
    return _zeta_floor(_heisenberg_clip(result))


def k_hydrolysis_effective(temp_C: float, pH: float) -> float:
    """Efektywna stała hydrolizy oligomeru — wzrasta z temperaturą i pH."""
    T_K = max(1.0, temp_C + 273.15)
    # pH optimum ok 7-8 dla RNA (Szostak 2018)
    ph_factor = 1.0 + 0.3 * abs(pH - 7.5)
    T_factor = math.exp(0.025 * (temp_C - 65.0))
    return K_HYDRO_BASE * T_factor * ph_factor


# ── Krok ewolucji puli oligomerów ─────────────────────────────────────────────

def step_oligomer_pool(
    state: EmergenceState,
    k_lig: float,
    k_hyd: float,
    dt: float,
    rng: np.random.Generator,
) -> None:
    """
    Jeden krok ewolucji rozkładu długości oligomerów.

    Procesy:
    1. Ligacja: oligomer(n) + monomer → oligomer(n+1)
       Rate: k_lig * counts[n] * monomer_pool
    2. Hydroliza: oligomer(n) → oligomer(n-k) + oligomer(k)
       Rate: k_hyd * counts[n] * (n-1)  [proporcjonalnie do liczby wiązań]
    3. GC drift: stochastyczna ewolucja składu GC (losowe mutacje punktowe)
    """
    pool = state.oligomer_pool
    N = pool.max_len
    counts = pool.counts.copy()
    mono   = pool.monomer_pool

    # Odnowienie puli monomerów: środowisko dostarcza monomery
    # (hydrotermalne wentyle, promieniowanie UV syntetyzuje nukleotydy)
    k_monomer_input = 5.0  # monomery/h — stały dopływ prebiologiczny
    mono = min(mono + k_monomer_input * dt, 2000.0)

    # Ligacja: oligomer(n) + monomer → oligomer(n+1)
    ligation_flux = np.zeros(N)
    total_ligation_loss = 0.0
    for n in range(N - 1):
        flux = k_lig * counts[n] * mono * dt
        flux = min(flux, counts[n])
        ligation_flux[n] -= flux
        if n + 1 < N:
            ligation_flux[n + 1] += flux
        total_ligation_loss += flux

    # Hydroliza: k_hyd maleje wykładniczo z długością >= 8 nt
    # Struktura drugorzędowa RNA (stem-loop) chroni wiązania
    # k_hyd_eff(L) = k_hyd * exp(-max(0, L-8)/20)  [Szostak 2018]
    hydro_flux = np.zeros(N)
    mono_return = 0.0
    for n in range(1, N):
        L = n + 1
        # Ochrona przez strukturę drugorzędową powyżej 8 nt
        struct_protection = math.exp(-max(0.0, L - 8.0) / 20.0)
        k_hyd_eff = k_hyd * struct_protection
        # Hydroliza proporcjonalna do liczby odsłoniętych wiązań (1 dla krótkich)
        exposed_bonds = max(1, min(L - 1, int(math.sqrt(L))))
        flux = k_hyd_eff * counts[n] * exposed_bonds * dt
        flux = min(flux, counts[n])
        hydro_flux[n] -= flux
        half = n // 2
        if half >= 1:
            hydro_flux[half] += flux * 0.5
            hydro_flux[max(0, n - half - 1)] += flux * 0.5
        else:
            mono_return += flux * L

    new_counts = counts + ligation_flux + hydro_flux
    # Heisenberg soft-clip: NaN/Inf → 0 (krótka ścieżka zaniku, nie katastrofa)
    new_counts = np.where(np.isfinite(new_counts), new_counts, 0.0)
    new_counts = np.maximum(new_counts, 0.0)
    pool.counts = new_counts
    pool.monomer_pool = max(0.0, _heisenberg_clip(mono - total_ligation_loss + mono_return))

    # GC drift — losowy spacer z słabym powrotem do optimum (chemiczna selekcja:
    # GC-rich RNA bardziej stabilne, przeżywa hydrolizę dłużej)
    gc_pull = 0.001 * (GC_OPT - state.gc_mean)  # Ornstein-Uhlenbeck powrót
    state.gc_mean = float(np.clip(state.gc_mean + gc_pull + rng.normal(0, 0.002), 0.25, 0.75))


def step_replication(
    state: EmergenceState,
    temp_C: float,
    dt: float,
    rng: np.random.Generator,
) -> None:
    """
    Krok replikacji szablonowej — logistyczny wzrost replikatorów.

    Pierwszy replikator pojawia się gdy oligomery >= L_MIN_REPLICATE
    z wystarczającą fitness (GC >= GC_MIN_STABLE).
    """
    pool = state.oligomer_pool
    n_above = float(pool.counts[L_MIN_REPLICATE - 1:].sum())

    # Warunek emergencji pierwszego replikatora
    if state.first_replicator_t is None and n_above > 1.0 and state.gc_mean >= GC_MIN_STABLE:
        # Stochastyczny próg: prawdopodobieństwo zapalenia replikatora
        p_emerge = min(0.95, n_above * 0.01 * dt)
        if rng.random() < p_emerge:
            state.first_replicator_t = state.t_h
            state.n_replicators = 1.0

    # Wzrost istniejących replikatorów (logistyczny)
    if state.n_replicators > 0:
        k_rep = K_TEMPLATE_BASE * math.exp(0.01 * (temp_C - 65.0))
        dn = k_rep * state.n_replicators * (1.0 - state.n_replicators / CARRYING_CAPACITY) * dt
        state.n_replicators = max(0.0, state.n_replicators + dn)


# ── Główna pętla symulacji ───────────────────────────────────────────────────

def simulate_first_rna(
    temp_C: float = 65.0,
    k_catalysis: float = 7.5,
    pH: float = 7.5,
    concentration_boost: float = 1000.0,
    drying_cycle_h: float = 12.0,
    drying_fraction: float = 0.3,
    hours: float = 500.0,
    dt_h: float = 0.5,
    topo_strength: float = 0.25,
    topo_pulsing: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> EmergenceState:
    """
    Symulacja powstawania pierwszego RNA w środowisku prebiologicznym.

    Parametry
    ----------
    temp_C       : temperatura środowiska [°C]
    k_catalysis  : stała katalityczna (minerał)
    pH           : pH środowiska
    hours        : czas symulacji [h]
    dt_h         : krok czasowy [h]
    topo_strength: siła pola topologicznego (Kähler)
    topo_pulsing : czy pole topologiczne pulsuje (True=akumuluje Berry)
    seed         : seed RNG

    Zwraca
    ------
    EmergenceState z pełną historią i czasem emergencji
    """
    from ..topology.fields import TopologyField
    from ..scenarios import ScenarioConfig, TopologyPattern, TimeDependence, SolventType

    rng = np.random.default_rng(seed)

    # Pomocniczy TopologyField dla Bloch coherence + Berry
    cfg_topo = ScenarioConfig(
        name="first_rna_topo", code="X", location="prebiotic",
        temp_C=temp_C, pressure_atm=1.0, UV_flux=20.0,
        solvent=SolventType.WATER, pH=pH, redox="mildly_reducing",
        energy_source="UV", k_energy=0.3,
        catalyst="clay", k_catalysis=k_catalysis, concentration_boost=500.0,
        k_synthesis=0.1, k_degradation=0.005,
        expected_protocells=100, timescale_description="hours",
        topo_strength=topo_strength,
        topo_pattern=TopologyPattern.SINUSOIDAL,
        topo_time_dependence=TimeDependence.PULSING if topo_pulsing else TimeDependence.STATIC,
        topo_pulse_freq=0.05, seed=seed,
    )
    topo = TopologyField(cfg_topo, Nx=32, Ny=32)

    state = EmergenceState(
        oligomer_pool=OligomerPool.seed(monomer_conc=800.0)
    )

    n_steps = int(hours / dt_h)
    k_hyd = k_hydrolysis_effective(temp_C, pH)

    record_every = max(1, int(2.0 / dt_h))

    if verbose:
        print(f"\n{'='*60}")
        print(f"FIRST RNA EMERGENCE SIMULATION")
        print(f"  Temp={temp_C}°C  pH={pH}  k_cat={k_catalysis}")
        print(f"  Topo_strength={topo_strength}  pulsing={topo_pulsing}")
        print(f"  Duration={hours}h  dt={dt_h}h")
        print(f"{'='*60}")

    for step in range(n_steps):
        state.t_h = step * dt_h
        topo.advance(state.t_h)

        bloch_c = topo.bloch_coherence()
        berry   = topo.berry_accumulated

        # Drying-wetting cycle (Damer & Deamer 2020, Sutherland 2016):
        # Cykl dobowy: ~30% czasu środowisko jest suche (odpływ pływowy)
        # W fazie suchej: hydroliza zatrzymana, ligacja 10x szybsza
        # (stężenie monomerów rośnie przez odparowanie wody)
        phase_in_cycle = (state.t_h % drying_cycle_h) / drying_cycle_h
        is_dry = phase_in_cycle < drying_fraction
        if is_dry:
            k_lig_eff = k_ligation_effective(
                temp_C, k_catalysis, bloch_c, berry, state.gc_mean,
                concentration_boost=concentration_boost * 10.0,  # odparowanie
            )
            k_hyd_eff = k_hyd * 0.02  # hydroliza prawie zerowa gdy sucho
        else:
            k_lig_eff = k_ligation_effective(
                temp_C, k_catalysis, bloch_c, berry, state.gc_mean,
                concentration_boost=concentration_boost,
            )
            k_hyd_eff = k_hyd

        step_oligomer_pool(state, k_lig_eff, k_hyd_eff, dt_h, rng)
        step_replication(state, temp_C, dt_h, rng)

        # Zapisz Berry w chwili emergencji
        if state.first_replicator_t is not None and state.berry_at_emergence is None:
            state.berry_at_emergence = berry

        # Historia
        if step % record_every == 0:
            n_above = state.oligomer_pool.n_above_threshold(L_MIN_REPLICATE)
            state.history['time_h'].append(state.t_h)
            state.history['mean_length'].append(state.oligomer_pool.mean_length())
            state.history['n_above_Lmin'].append(n_above)
            state.history['n_replicators'].append(state.n_replicators)
            state.history['berry'].append(berry)
            state.history['gc_mean'].append(state.gc_mean)
            state.history['k_lig_eff'].append(k_lig_eff)

        if verbose and step % (record_every * 20) == 0:
            n_ab = state.oligomer_pool.n_above_threshold(L_MIN_REPLICATE)
            rep_str = f"REPLIKATOR@{state.first_replicator_t:.1f}h" if state.first_replicator_t else "brak"
            print(f"  t={state.t_h:7.1f}h | "
                  f"<L>={state.oligomer_pool.mean_length():.1f}nt | "
                  f"N>={L_MIN_REPLICATE}={n_ab:.0f} | "
                  f"repl={rep_str} | "
                  f"berry={berry:.4f} | "
                  f"k_lig={k_lig_eff:.2e}")

    if verbose:
        if state.first_replicator_t is not None:
            print(f"\n✓ PIERWSZY REPLIKATOR: t = {state.first_replicator_t:.1f} h")
            print(f"  Berry phase w chwili emergencji: {state.berry_at_emergence:.5f} rad")
            print(f"  Replikatorów końcowo: {state.n_replicators:.1f}")
        else:
            print(f"\n✗ Brak emergencji w {hours}h")

    return state


# ── Scan predykcyjny ─────────────────────────────────────────────────────────

def scan_emergence_conditions(
    temp_range: tuple = (20.0, 100.0),
    n_temp: int = 6,
    k_cat_range: tuple = (2.0, 15.0),
    n_kcat: int = 4,
    hours: float = 800.0,
    dt_h: float = 1.0,
    seed: int = 0,
    concentration_boost: float = 1000.0,
    drying_cycle_h: float = 12.0,
    drying_fraction: float = 0.4,
    topo_strength: float = 0.5,
    topo_pulsing: bool = True,
) -> list[dict]:
    """
    Scan parametrów środowiskowych → mapa predykcji czasu emergencji.

    Zwraca listę słowników z wynikami dla każdej kombinacji parametrów.
    """
    results = []
    temps = np.linspace(*temp_range, n_temp)
    kcats = np.linspace(*k_cat_range, n_kcat)

    total = n_temp * n_kcat
    done  = 0
    print(f"\nPREDYKCJA: {total} kombinacji parametrów...")

    for temp in temps:
        for kcat in kcats:
            state = simulate_first_rna(
                temp_C=temp, k_catalysis=kcat,
                hours=hours, dt_h=dt_h,
                concentration_boost=concentration_boost,
                drying_cycle_h=drying_cycle_h,
                drying_fraction=drying_fraction,
                topo_strength=topo_strength,
                topo_pulsing=topo_pulsing,
                seed=seed, verbose=False,
            )
            results.append({
                'temp_C': float(temp),
                'k_catalysis': float(kcat),
                'T_emergence_h': state.first_replicator_t,
                'berry_at_emergence': state.berry_at_emergence,
                'final_replicators': state.n_replicators,
                'emerged': state.first_replicator_t is not None,
            })
            done += 1
            if done % 4 == 0:
                print(f"  {done}/{total} done...")

    return results
