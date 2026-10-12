# Note: the source file of an index run

Date: 2026-10-12
Status: applies from the `cut-pass` branch.

## The change

Task 4 of `docs/superpowers/plans/2026-10-12-cut-pass.md` changed one
signature:

```rust
pub fn IndexRun::start_with(video: &Path, plan: &Plan) -> Result<Self>          // before
pub fn IndexRun::start_with(video: &Path, plan: &Plan, media: &Path) -> Result<Self>  // after
```

The third argument is the store media folder. The run puts the progressive copy
of a fragmented MP4 there.

The run holds the file to cut in a `source` field. The `label` field holds the
path of the user, and the `source` field holds the file that ffmpeg reads. The
two differ only for a fragmented MP4.

`IndexRun::start(video)` still takes one argument, because a plain video is its
own source.

## Why the run holds two paths

The label names the video in the store, in the result list, and in the file
list. The user knows that name. The source names the bytes on disk. A rewrite
must not change the name of a stored record, because a search answers with the
label.

## What a future change touches

| Change | Files |
| --- | --- |
| A new source rule, for example a network path or a second copy form | `media::source_file` |
| A new way to name a copy | `media::copy_name` |
| A removal of the copy after the run | `IndexRun::clear_dir`, plus the store media folder |
| A plan that re-cuts an indexed video | the `source` field stands. The copy stands while it is newer than the video. |

## The limit today

The copy stays on disk after the run, and the app never removes it. A video that
moves or changes makes the old copy stale. The next run makes a new copy, and
the old file stays. A cleanup pass waits for a later plan.

## Where the copy sits

```text
<store folder>/
  index.db
  media/
    <source name>_<hash of the full path>.mp4
```

The hash keeps two videos with one name apart. The name of the source stays
readable, so a hand check is easy.
