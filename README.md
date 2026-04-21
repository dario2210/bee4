# bee4

`bee4` to pierwsza wersja strategii WaveTrend zbudowana na tej samej strukturze projektu co `bee1`.
Dashboard, wykres, flow backtest/WFO/live oraz uklad plikow pozostaja spojne z wczesniejszym projektem, ale logika wejsc i wyjsc zostala przestawiona na sygnaly WaveTrend.

## Co robi projekt

- domyslnie pracuje w trybie `long/short` z lustrzana logika po obu stronach rynku
- otwiera `long` na zielonej kropce WaveTrend pod zerem oraz w oknie kilku barow po tym sygnale
- otwiera `short` na czerwonej kropce WaveTrend nad zerem oraz w analogicznym oknie kilku barow po tym sygnale
- dla `long` wymaga odzyskania `EMA20`, a dla `short` odrzucenia `EMA20`, zeby odsiac slabsze setupy
- dopuszcza re-entry, gdy WaveTrend nadal utrzymuje sie blisko zera po ostatniej zielonej lub czerwonej kropce
- pozycja zamyka sie i odwraca dopiero na przeciwnym sygnale, zgodnie z profilem WaveTrend
- wspiera backtest, walk-forward optimization oraz live/paper runner
- zachowuje dashboard z wizualizacja ceny, markerow transakcji oraz panelu `WT1/WT2`

## Parametry WFO

W pierwszej wersji strategii testowane sa:

- `wt_channel_len`
- `wt_avg_len`
- `wt_signal_len`
- `wt_min_signal_level`

Domyslna siatka WFO:

- `channel_len`: `8, 10, 12, 14`
- `avg_len`: `14, 21, 28, 35`
- `signal_len`: `3, 4, 5`
- `min_signal_level`: `0, 5, 10, 20`

W tej wersji WFO optymalizuje glownie rdzen WaveTrend, a lustrzane filtry wejsc
`EMA20 + re-entry window` pozostaja na stalych, recznie dobranych wartosciach.

## Najwazniejsze pliki

- [bee4_main.py](bee4_main.py)
- [bee4_dashboard.py](bee4_dashboard.py)
- [bee4_strategy.py](bee4_strategy.py)
- [bee4_wfo.py](bee4_wfo.py)
- [bee4_live_runner.py](bee4_live_runner.py)

## Uruchomienie

```bash
python bee4_dashboard.py --host 0.0.0.0 --port 8064
```

```bash
python bee4_main.py --mode backtest
```

```bash
python bee4_main.py --mode wfo
```
