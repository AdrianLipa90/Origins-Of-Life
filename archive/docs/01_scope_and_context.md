# Rozdział 1: Zakres i Kontekst Projektu

## 1.1 Czym jest Origins of Life / NOEMA

Projekt **Origins of Life** (w skrócie OoL) to symulacyjno-teoretyczne badanie emergence życia z materii nieożywionej (abiogeneza) z wykorzystaniem geometryczno-fazowego frameworku CIEL/NOEMA. Powstał jako rozszerzenie systemu CIEL/0 — Consciousness-Integrated Emergent Lattice — na problem powstania pierwszych replikatorów biologicznych.

Projekt obejmuje:
- Symulację emergence DNA z prostych prekursorów chemicznych
- Modelowanie protokomórek (protocell) z dynamiką klimatyczną i holonomią fazową
- Orbitalno-semantyczną kompresję procesów biochemicznych na role NOEMA (I/M/A/AT/F)
- Synchronizację Kuramoto oscylatorów fazowych jako model sprzężenia zwrotnego między chemią a geometrią

## 1.2 Struktura archiwum

Master ZIP zawiera 23 wewnętrzne archiwa podzielone na 5 kategorii:

| Folder | Zawartość | Liczba plików |
|--------|-----------|---------------|
| `01_versions` | Historyczne snapshoty v03–v10 + Origins-Of-Life-main + planetary biology | 1321 |
| `02_results` | Wyniki symulacji (RESULTS_ONLY) — DNA four blocks v05, DNA to protocell v06 | 15 |
| `03_canon_solver` | Merge kodu, kotwice powierzchniowe, krystalizacja wyników wyszukiwania | 1663 |
| `04_latex` | Zgromadzone pliki TeX/Bib/Class z wszystkich archiwów | 160 |
| `05_archive_failures` | non_arbitrary_pass_without_closure v10 — brak domknięcia semantycznego | 54 |

## 1.3 Pochodzenie danych

Archiwa zostały zrekonstruowane ze śladów NOEMA (trace) z lokalnego systemu plików `/mnt` na hostie NOEMA surface. Wersje v03–v10 są rekonstrukcjami, a nie byte-exact kopiami oryginalnych repozytoriów. Każdy katalog zawiera `RECONSTRUCTION_MANIFEST.json` dokumentujący provenance i znane braki.

## 1.4 Cel dokumentacji

Celem niniejszej dokumentacji jest:
1. Przedstawienie założeń teoretycznych leżących u podstaw symulacji
2. Opisanie założeń komputacyjnych i architektury kodu
3. Udokumentowanie chronologii — co, kiedy i dlaczego powstało
4. Analiza wyników każdej wersji
5. Określenie statusu domknięcia semantycznego projektu
