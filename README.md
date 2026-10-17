# Aiva studio

Makes the photoreal personas for **Aiva**, the assistant on
[BlissPoint Access](https://blisspointaccess.com). It runs only open-source models, on
free GitHub runners. Every person here is fictional: invented from a written casting
brief (`people.json`), never based on a real person.

## How it works

1. **Portraits.** RealVisXL V5.0, an open photoreal model, runs on the CPU through
   stable-diffusion.cpp. It makes three candidates per persona in `out/candidates/<id>/`.
2. **Pick.** Write the chosen candidate into `picks.json`, for example `{ "sienna": 2 }`.
3. **Animate.** LivePortrait brings the chosen portrait to life. It's driven by explicit
   motion (head turns, nods, tilts, blinks, speech, and smile, wink and laugh
   expressions) rather than by a video of anyone. The result is about eleven short clips
   per persona in `out/people/<id>/`. Every clip starts and ends on the portrait itself,
   so the website can chain them seamlessly. Review contact sheets go in
   `out/review/<id>/`.

## Real motion

The personas can move like a real person instead of following the written choreography.
On the website, an admin opens **Admin → Aiva Motion** (`/admin/aiva-motion`), sits in
front of their webcam and follows a 96-second on-screen script: sit, glance away, listen,
talk, think, nod, smile, laugh, tilt the head, wink. MediaPipe's face tracker runs in
their browser and records only numbers: head angles, blinks, mouth opening and smile, 25
times a second. No video or picture of the person is uploaded anywhere, and the personas
keep their own faces.

Upload the downloaded file into `driving/` (GitHub: Add file → Upload files → Commit).
That push alone starts a run that re-animates every persona with it (`studio/drive.py`);
the recorded clips are 4–12 seconds long instead of 4–5. Clips the recording doesn't
cover keep the written choreography. It takes about four hours on the free runners. Then
bring the media into the site with `node scripts/aiva-avatars/generate.mjs import` in the
web repo.

To try a recording on one persona first, set `request.json` to
`{"step": "animate", "only": ["sienna"], "motion": "driving/<file>.json", "preview": true}`.

## Running it

Edit `request.json` and push it; `.github/workflows/studio.yml` does the rest and
commits results back. The fields are:

- `step`: `portraits` or `animate`
- `only`: persona ids; leave it empty for all twelve
- `count`: number of candidates per persona
- `model`: `lightning` (fast) or `full`
- `clips`: a subset of clips to (re)make
- `motion`: a motion recording to move like (`driving/<file>.json`); see Real motion
- `preview`: write to `out/preview/<id>/` and leave `out/people/` alone

You can also start it from Actions → Studio → Run workflow.

## Credits and licences

See [NOTICE.md](NOTICE.md).
