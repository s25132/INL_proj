# Przywracanie interpunkcji w tekstach polskich

Projekt z Inżynierii Lingwistycznej porównujący modele BiLSTM, HerBERT i RoBERTa w zadaniu automatycznego przywracania interpunkcji.

## Opis zadania

Celem jest uzupełnienie znaków interpunkcyjnych w polskim tekście, z którego usunięto interpunkcję. Takie rozwiązanie może poprawiać czytelność transkrypcji automatycznego rozpoznawania mowy (ASR). Zadanie jest inspirowane konkursem [PolEval 2021 — Punctuation Restoration](https://poleval.github.io/2021-punctuation-restoration/).

Problem został ujęty jako klasyfikacja sekwencyjna: dla każdego słowa model przewiduje znak, który powinien wystąpić bezpośrednio po nim. Obsługiwanych jest sześć klas:

| Etykieta | Znaczenie |
| --- | --- |
| `BRAK` | Brak znaku interpunkcyjnego |
| `.` | Kropka |
| `,` | Przecinek |
| `?` | Pytajnik |
| `!` | Wykrzyknik |
| `:` | Dwukropek |

Przykładowe oczekiwane przekształcenie:

```text
Wejście: Kiedy wrócisz daj mi znać
Wyjście: Kiedy wrócisz, daj mi znać.
```

Implementacja zachowuje wielkość liter w słowach i nie uczy się jej przywracania. Zakres etykiet jest uproszczony względem oryginalnego zadania PolEval: myślnik i wielokropek nie mają osobnych klas.

## Dane i przygotowanie eksperymentu

Notatniki łączą trzy źródła tekstów:

- `2021-punctuation-restoration-main/train/expected.tsv` — teksty referencyjne z interpunkcją ze zbioru PolEval 2021. Ścieżkę można zmienić przez `DATA_DIR`.
- `polish-news.csv` — lokalny plik danych ze zbioru [WiktorS/polish-news](https://huggingface.co/datasets/WiktorS/polish-news/tree/main); wykorzystywana jest kolumna `content`. W repozytorium źródłowym plik nosi nazwę `full_data.csv`.
- `zdania_polaczone_4000.csv` — dodatkowy lokalny zbiór zdań, również wczytywany z kolumny `content`, wskazany przez `EXCLAMATION_PATH`.

W aktualnym kodzie wszystkich czterech notatników `NEWS_LIMIT = 10000`, a `EXCLAMATION_LIMIT = None` oznacza wczytanie całego dodatkowego pliku. Wiążące są wartości w komórkach kodu; niektóre opisy w notatnikach wspominają o 100 000 rekordów Polish News.

Słowa są wyodrębniane wyrażeniem regularnym `\w+`, a etykiety powstają na podstawie interpunkcji między kolejnymi słowami. Połączone dokumenty są losowo dzielone na trening, walidację i test w proporcji 80/10/10, z ziarnem `SEED = 42`. Podział odbywa się przed tworzeniem fragmentów tekstu dla transformerów. Słownik BiLSTM jest budowany wyłącznie na danych treningowych.

## Opis notatników

Każdy notatnik zawiera wczytanie i przygotowanie danych, trening modelu, ocenę na zbiorze testowym, predykcję dla własnego zdania oraz zapis modelu.

| Notatnik | Model i sposób działania | Zapis modelu |
| --- | --- | --- |
| [PolEval_2021_przywracanie_interpunkcji_BiLSTM.ipynb](PolEval_2021_przywracanie_interpunkcji_BiLSTM.ipynb) | Model z warstwą embedding, dwuwarstwowym dwukierunkowym LSTM i klasyfikatorem przewidującym etykietę po każdym słowie. Stanowi punkt odniesienia dla transformerów. | `models/bilstm/checkpoint.pt` |
| [PolEval_2021_przywracanie_interpunkcji_HerBERT.ipynb](PolEval_2021_przywracanie_interpunkcji_HerBERT.ipynb) | Dostrajanie `allegro/herbert-base-cased` do klasyfikacji tokenów. Etykieta słowa jest przypisana do jego pierwszego tokenu podrzędnego; pozostałe są pomijane przy obliczaniu straty. | `models/herbert/` |
| [PolEval_2021_przywracanie_interpunkcji_RoBERTa.ipynb](PolEval_2021_przywracanie_interpunkcji_RoBERTa.ipynb) | Dostrajanie `sdadas/polish-roberta-base-v2` z analogicznym przypisaniem etykiet do pierwszych tokenów słów. | `models/roberta2/` |
| [PolEval_2021_przywracanie_interpunkcji_RoBERTa_best.ipynb](PolEval_2021_przywracanie_interpunkcji_RoBERTa_best.ipynb) | Zachowany wariant notatnika RoBERTa, mającego najlepsze wyniki. | `models/roberta/` |

HerBERT i oba warianty RoBERTa dzielą dokumenty na fragmenty do 96 słów, z limitem 256 tokenów modelu i dalszym podziałem zbyt długich fragmentów. Do treningu wykorzystano ważoną stratę `CrossEntropyLoss`, aby zwiększyć wpływ rzadszych klas. Mechanizm early stopping wybiera wagi na podstawie straty walidacyjnej.

## Ocena modeli

Notatniki obliczają precision, recall i F1 dla poszczególnych klas oraz Macro F1 i Weighted F1. Ocena odbywa się na poziomie słów. Wyniki dotyczą własnego zbioru testowego wydzielonego z połączonych danych, a nie oficjalnego testu ani oficjalnej procedury punktacji PolEval.

## Uruchomienie

1. Przygotuj środowisko Pythona z pakietami z `requirements.txt` oraz Jupyter Notebook lub obsługą notatników w VS Code. Wpis PyTorch w tym pliku jest przeznaczony dla Windows, Pythona 3.12 i CUDA 12.6; dla innej konfiguracji wymaga dostosowania.
2. Uruchom wybrany notatnik z katalogu projektu. Sprawdź `DATA_DIR`, `NEWS_PATH` i `EXCLAMATION_PATH` oraz obecność kolumny `content` w plikach CSV.
3. W razie potrzeby dostosuj limit danych, liczbę epok i rozmiar batcha. Modele automatycznie korzystają z CUDA, jeśli jest dostępna, albo z CPU. Przy pierwszym uruchomieniu transformerów pobierane są wagi i tokenizery z Hugging Face.
4. Wykonaj komórki kolejno od początku. Końcowe komórki pozwalają sprawdzić predykcję dla własnego tekstu i zapisać model.

## Źródła

- [WiktorS/polish-news — Hugging Face](https://huggingface.co/datasets/WiktorS/polish-news) — źródło korpusu polskich wiadomości używanego w projekcie.
- [PolEval 2021 — Punctuation Restoration](https://poleval.github.io/2021-punctuation-restoration/) — opis zadania przywracania interpunkcji oraz zbioru WikiPunct.
- [Repozytorium danych PolEval 2021](https://github.com/poleval/2021-punctuation-restoration) — dane i dokumentacja zadania; lokalna kopia znajduje się w `2021-punctuation-restoration-main/`.
