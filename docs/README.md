# Factor network visualization

Interactive 3D view of the eight-factor EFA solution: factor–feature loadings, search/filter, and example melodies with an in-browser piano-roll player.

This folder is self-contained static content. No build step or package install is required.

## Run locally

1. Unzip this folder (if needed) and open a terminal **inside** it (the directory that contains `index.html`).
2. Start a local web server:

   ```bash
   python3 -m http.server 8000
   ```

   On Windows, `py -m http.server 8000` also works if Python is installed.
3. In a browser, open: [http://localhost:8000/](http://localhost:8000/)

Stop the server with `Ctrl+C` in the terminal.

### Notes

- Do **not** open `index.html` via `file://` (double-clicking the file). The page loads data and MIDI examples over HTTP; `file://` will break those requests.
- An internet connection is needed for the graph and piano-roll libraries (loaded from public CDNs). All analysis data and MIDI files are included locally under this folder.
- If port 8000 is already in use, choose another port, e.g. `python3 -m http.server 8080`, then open `http://localhost:8080/`.

## Using the page

- Drag to rotate the network; scroll to zoom.
- Click a **factor** or **feature** node to open its panel (loadings and high/low example melodies where available).
- Use search and the loading-threshold control to focus the graph.
- In the melody panel, use the tab and ‹ › controls to browse examples; the piano roll supports scroll, zoom, and playback.
