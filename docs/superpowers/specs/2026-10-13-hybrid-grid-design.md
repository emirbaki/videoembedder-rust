# Both grids for one video - design

Date: 2026-10-13
Status: draft. The user settled four decisions on 2026-10-13.

Sources:

- `docs/deferred-work.md` section 1.1 holds the item.
- `docs/google-gallery-comparison.md` line 224 holds the Google note.
- `docs/superpowers/specs/2026-10-12-cut-batch-search-design.md` section 2
  holds decision 2, which this spec reverses.
- `videoembed/src/plan.rs`, `videoembed/src/index.rs`,
  `videoembed/src/store.rs`, `videoembed/src/db.rs`,
  `videoembed/src/bin/web.rs`, and `videoembed/assets/app.js` hold the code.

## 1. The goal

One video answers two questions together. A window record answers "what happens
here". A frame record answers "which second". Today a plan holds one mode, so a
video holds one grid, and a plan change drops the records of the other.

## 2. The decisions

The user settled these four on 2026-10-13:

1. A plan flag holds the choice. The window plan stays as it is, and a `fine`
   flag adds the 0.25 s grid beside it.
2. One search ranks both kinds. The hits fold at `MERGE_GAP`, and every hit
   keeps its kind.
3. The speed report splits the cold load from the warm run, and it keeps the
   last runs as a list. That work gets its own small change.
4. The page skips a model load when the mode is already active. That work gets
   its own small change.

### 2.1 The reversal

The cut, batch, and search spec dropped this item. Its reason was that two
grids double the records of a video, and that one result list would mix two
score scales. The user asks for the item now. Two facts answer the reason:

- The code already ranks every record kind in one list. `search_rows`
  (index.rs:935) puts a window record, a frame record, and an audio record of
  the scope through the same ranking, the same baseline, and the same gap. The
  two grids need no new merge rule.
- Both kinds come from one model and one query space. The baseline is the
  median of the scope (index.rs:44), and the margin is the score over the
  baseline. A mixed scope gives one middle for both kinds.

The double count stands. Section 8 measures it.

## 3. The plan flag

`Plan` (plan.rs:70) gains one field:

```rust
/// The plan also cuts the 0.25 s fine grid.
pub fine: bool,
```

The four presets set it false, so the default plan does not change.

`set_words` (plan.rs:153) gains the key `fine=on|off`. The key sets the flag
and leaves the window values alone. The page and the console both reach the
plan through this function, so one key serves both.

## 4. The store

The `videos` table (db.rs:19) gains one column:

```sql
plan_fine INTEGER NOT NULL DEFAULT 0
```

The migration (db.rs:85) checks for `plan_mode` alone and returns when it finds
it. A store from before this change holds `plan_mode` and no `plan_fine`, so
the check must move to one helper for each column. `set_plan` (store.rs:347)
and `plan_of` (store.rs:310) carry the flag.

## 5. One cut, two grids

`cut_media` (index.rs:659) cuts every frame of the video one time, at the rate
of the plan. `frames_in` (index.rs:219) and `window_times` (index.rs:233) index
that cut by the plan rate, so the cut rate and the plan rate must match today.

A plan with the flag cuts at the denser of the two rates. The fine grid needs
4 fps (`FRAME_RATE`, index.rs:26). The window path then picks its own frames
out of the dense cut:

- `frames_in` takes the cut rate beside the plan rate, and it takes every
  `cut_rate / plan_rate` frame of a window.
- `window_times` names each picked frame at its own time in the cut.

An Object plan measures 1 fps, so the dense cut holds four frames for each one
the window path wants. A plan whose own rate is already 4 fps or more keeps its
own rate.

## 6. One index pass

One run writes both sets. The window phase writes the window records as it does
today. The fine phase reads the frames of the dense cut at 0.25 s spacing,
whatever the rate of the cut. It writes one frame record for each of those
frames, and one 1 s audio record for each second.

The fine grid is fixed. It is the Fine preset (plan.rs:91): a 0.25 s window, a
0.25 s stride, one frame, and audio on. The flag adds that grid. The flag never
changes it.

One job, one progress bar, and one done mark cover both sets.

## 7. One search

No merge rule changes. One addition: `Hit` and `Row` (index.rs:110 and 145)
gain the record kind, so the page can mark a window hit apart from a frame hit.
The bar already takes the frames of a video when the video holds them
(index.rs:895).

## 8. The cost

A 30 s video with the Object plan writes 29 window records today. The same
video with the flag writes 29 window records, about 120 frame records, and
about 30 audio records. The run costs about 150 model calls against 29.

The fine grid is the expensive part. A user who wants speed leaves the flag
off.

## 9. The page

The plan panel gains a "Both grids" checkbox beside the audio box. A moved
slider clears the checkbox, as it clears a preset. The plan body (web.rs:171)
gains a `fine` field. A hit row shows its kind, `window` or `frame`.

## 10. The tests

- The flag round-trips through the store.
- `set_words("fine=on")` sets the flag and leaves the window values.
- The window path picks the right frames out of a dense cut.
- A search over a window record and a frame record of one video folds them at
  the gap when they sit inside it, and keeps both when they sit apart.
- A hit carries its kind.
- The page shows the checkbox, and the body carries the field.

## 11. What this spec does not cover

- The cold and warm speed report. It gets its own small change.
- The reload guard. It gets its own small change.
- The audio records of the window plan. The window record holds its own window
  audio, and the fine grid holds its own 1 s records. This spec keeps both.
- A store over one million records. Section 3 of `docs/deferred-work.md` holds
  that work.
