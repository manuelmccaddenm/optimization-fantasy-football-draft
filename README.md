# optimization-fantasy-football-draft

El *draft* de fantasy football planteado como una sucesión de programas
cuadráticos con restricciones (optimización de Markowitz media y varianza),
con dos estrategias opuestas:

- **Piso** (Markowitz): minimiza la varianza; programa cuadrático convexo
  resuelto con **Punto Interior**.
- **Techo** (Upside): premia la varianza; problema no convexo resuelto con
  **Descenso de Gradiente Proyectado** (la proyección es, a su vez, un QP
  convexo resuelto con el mismo Punto Interior).

## Estructura

- `optimizacion.py` — algoritmos: Punto Interior, Descenso de Gradiente
  Proyectado, reemplazo dinámico, factibilidad de ADP y el simulador del
  draft serpiente.
- `construir_datos.py` — lee `datos_crudos/` (nflverse 2021-2024, ADP 2025,
  pateadores, novatos, lesiones) y escribe `data/jugadores.csv`,
  `data/mu.csv`, `data/sigma.csv`.
- `cuaderno.ipynb` — carga esos CSV, importa `optimizacion` y corre los
  experimentos (un draft, tablero serpiente, Monte Carlo, plano riesgo y
  rendimiento). No contiene algoritmos ni la construcción de matrices.
- `report.pdf` — el informe.

## Reproducir

```bash
pip install -r requirements.txt
python construir_datos.py        # genera data/*.csv
jupyter notebook cuaderno.ipynb  # corre los experimentos y guarda figs/
```

`data/*.csv` ya viene incluido, así que el cuaderno corre sin volver a
ejecutar `construir_datos.py`.
