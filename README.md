# Rainbow Rows

A one-touch puzzle game: drop colored tiles, make runs of 5 consecutive tiles
showing all five colors (across or down) to clear them. Clear enough runs to
beat each round before the board fills. Rounds get faster, and from round 3
the board starts pre-seeded with junk.

**Play:** enable GitHub Pages on this repo, then open the URL on your phone.
Tip: "Add to Home Screen" in your browser menu gives it an app icon.

Built as a single self-contained `index.html` — no build step, no dependencies.
Designed to be wrapped with Capacitor for a native Android/iOS build later.

## Dev log
- v1–v2: full-row rainbow rule, 10x10, turn timer with auto-drop
- v3: run-of-6 expansive clears, double bag
- v4: 9x9, 5 colors, rounds with fanfare, junk seeding
- v5: removed hint markers (playtest: harmful), full-width grid
- v6: disintegration animation on clears; detection logic fuzz-tested
