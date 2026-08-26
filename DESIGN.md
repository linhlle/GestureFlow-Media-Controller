# Design

Notes on why the site looks the way it does, and what it looked like before.

---

## Part 1 — what read as machine-generated

Audited `web/css/style.css` (486 lines) and the four pages before changing
anything. These are the specific tells, with what was actually in the files.

### The palette was the default palette

```css
--bg:     #0d1117;   /* GitHub dark's background, verbatim */
--accent: #4dd4c4;   /* teal */
```

Teal on blue-black slate. It is the single most common colour pair in generated
UI, and `#0d1117` is not a chosen colour — it is the one that comes out when
nobody chose. Every accent in the file descends from it: `--accent-dim` is the
same hue darkened, the mode pills, the progress bars, the skeleton overlay, the
list bullets, the step numerals.

### One typeface, and it was nobody's

```css
--sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, system-ui, sans-serif;
```

System sans for everything, with Inter named in the stack — the default of
defaults. Worse, the headings asked for weights the stack does not have:

```
font-weight: 620    font-weight: 640    font-weight: 650    font-weight: 680
```

Four bespoke-looking weights, none of which exist in a system font. The browser
either synthesises them or snaps to 700, so the careful-looking numbers did
nothing at all. That is a tell in itself: it looks like typographic intent
without being any.

### Everything was the same rounded box

`border-radius: var(--radius)` (10px) appeared on 18 declarations, and this
exact recipe — hairline border, raised fill, 10px radius — repeated 7 times
under different names:

```css
border: 1px solid var(--border);
background: var(--bg-raised);
border-radius: var(--radius);
```

`.split-card`, `.panel`, `.binding`, `.stage`, `pre`, `.preview`, the tables.
A landing-page feature card and a live inference readout are not the same kind
of object, and they should not look identical. Plus `999px` pills and two
`border-radius: 50%` circles. Nothing in the design had a corner.

### Glassmorphism nav

```css
backdrop-filter: blur(8px);
background: rgba(13, 17, 23, 0.92);
```

The sticky translucent blurred header, shipped by default in roughly every
generated site since 2022.

### The copy was written by nobody

Third person, evenly weighted, no opinions:

> "GestureFlow tracks one hand through your webcam and turns it into cursor
> movement, clicks, scrolling, volume, and keyboard shortcuts."

Accurate, and it could be describing anyone's project. Nothing in the voice
suggested a person made choices, got something wrong, or cared.

### Two things it was doing right, and I kept them

To be fair to the file: there were **no gradient blobs, no glow effects, and no
gradient-filled headline** — the usual suspects were absent. And the
`.notice` disclaimer about what a web page cannot do was already honest. That
stays, reworded but not softened.

### And two real accessibility bugs

```css
input:focus, select:focus, textarea:focus {
  outline: none;               /* removed, never replaced */
  border-color: var(--accent-dim);
}
```

`outline: none` with no `:focus-visible` alternative anywhere in the file —
zero occurrences. Keyboard users lost the focus ring on every form control in
the command builder, and a 1px border tint is not a focus indicator.

```css
html { scroll-behavior: smooth; }
```

Forced on everyone, with no `prefers-reduced-motion` block anywhere in the
stylesheet. Someone who has asked their OS to stop animating things still got
animated scrolling.

---

## Part 2 — two directions

The brief left the taste section blank, so here are two, with a default
chosen and built.

### A — Field notes  ← **built**

The project's whole character is measurement: `DIAGNOSIS.md` argues from
numbers, the README keeps a measured-vs-not-measured table, and the site's
honest bit is a disclaimer about what software cannot do. So: a lab notebook.
Warm paper, ink, one saturated red for anything that matters, hairline rules
instead of boxes, and mono annotations in the margin like figure numbers.

It is also the bigger departure. Light-on-warm-paper is genuinely rare in
generated output, which trends dark-slate almost without exception.

**Palette** — warm neutrals, one hot accent:

| Token | Value | Use |
| :--- | :--- | :--- |
| `--paper` | `#f6f2e9` | ground |
| `--paper-2` | `#efe9dc` | sunken fills, code |
| `--ink` | `#1c1a16` | body text, rules that matter |
| `--ink-2` | `#4d4840` | secondary text |
| `--ink-3` | `#6f685c` | captions, marginalia |
| `--rule` | `#d5ccba` | hairlines |
| `--vermillion` | `#c2402a` | links, brand, the hand skeleton |
| `--ochre` | `#8a6a12` | warnings |
| `--moss` | `#3f6b3a` | valid state |
| `--brick` | `#8f1d1d` | errors |
| `--plate` | `#17150f` | the camera stage — warm near-black, not blue |

**Type** — Fraunces + IBM Plex, self-hosted:

- **Fraunces** (variable, `opsz` 9–144) for display. A soft-serif with real
  personality — it gets wonkier and more editorial at large optical sizes,
  which is exactly what a headline wants and exactly what Inter cannot do.
- **IBM Plex Sans** for text. Humanist but drawn by engineers; it sits next to
  a warm serif without arguing with it.
- **IBM Plex Mono** for code, readouts, and marginalia. Same family, so the
  technical furniture belongs to the same voice as the prose.

152 KB total, latin subset, `woff2`, served from `web/fonts/`. No third-party
font request at runtime.

**Signature detail** — the hand itself. The 21 MediaPipe landmarks and the same
connection topology `demo.js` draws, rendered as an ink sketch: small in the
masthead, large as a hero figure that draws itself in once on load. It is the
one mark that could only belong to this project.

### B — Darkroom (not built)

If the site must stay dark: warm charcoal `#131210` rather than blue slate,
with high-chroma amber `#e8a33d` and a magenta secondary `#c8437f`. Monospace
led — IBM Plex Mono for headings as well as code, with **Space Grotesk** for
body. Zero radius, 2px rules instead of hairlines, deliberate 8px-grid
misalignment. Brutalist and loud where A is quiet.

Rejected as the default only because it keeps the dark background the current
site already has, so it would read as a re-skin rather than a decision. The
palette tokens are structured so switching is a token swap, not a rewrite.

---

## Part 3 — rules the redesign follows

- **Rules, not boxes.** A hairline or a change of ground separates things.
  Borders on four sides are reserved for objects that are genuinely contained:
  the camera stage, form controls.
- **Hard edges.** `--radius: 2px`, and only on things you type into. Nothing
  floats, so nothing needs a soft shadow. Where depth is wanted, it is a hard
  offset in ink — a printed register mark, not a blur.
- **One accent, used sparingly.** Vermillion means *interactive or important*.
  If everything is red, nothing is.
- **The margin rail carries mono.** Section numbers, captions, and asides live
  in the left rail in Plex Mono at 0.72rem. Prose never does.
- **Measure is capped at 68ch.** Long lines are the fastest way to make a page
  feel unconsidered.
- **Focus is visible and vermillion.** `:focus-visible` with a 2px outline and
  3px offset, everywhere, including the controls that had `outline: none`.
- **`prefers-reduced-motion` turns everything off**, including the hero draw-in
  and the smooth scrolling that was previously unconditional.
