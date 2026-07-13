# Rainbow Games

A small arcade of original puzzle games. Single self-contained HTML files,
no dependencies, no build step. Live via GitHub Pages.

- **index.html** — game picker menu
- **rows.html** — Rainbow Rows: drop tiles, line up all five colors to clear
- **silt.html** — Rainbow Silt: falling-sand puzzle; bridge the walls with one
  color to wash it away, light up all five colors to win the round

## Dev log
- Rows v1–v8: full-row rule → run-of-5 expansive clears → 9x9/5-color rounds,
  timer disc, deal-in animation, fuzz-tested clear detection
- Silt v1: cellular sand sim (typed arrays, ~9k cells), bridge detection via
  BFS, instant-clear-on-connection, stone terrain from round 2. Tuned
  field-to-blob ratio so two same-color pours can bridge.
