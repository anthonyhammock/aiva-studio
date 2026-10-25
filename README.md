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

## Warmth and long takes

Every clip starts and ends on the same resting pose: the portrait wearing the persona's
soft, closed-mouth smile (`WARMTH` in `studio/motions.py`; Rosa and Grace were cast
smiling, so theirs is small or none). A neutral resting face reads as cool or stern on a
live call. The smile is drawn on LivePortrait's expression keypoints (`KP_SMILE`) and
mixed with the laugh template's (`REST_MIX`); smiles inside clips use the same shape, so
a full smile peaks at the same strength whatever the warmth. Changing a persona's warmth
remakes all of its clips on the next run (each clip records the warmth it was made with).

To choose the warmth by eye, run `{"step": "animate", "clips": ["warmth_sheet"],
"preview": true}`: each persona is rendered at a range of smiles into
`out/preview/<id>/review/warmth.jpg` in a few minutes, without remaking any clip.

Besides the 4–5 second clips there are three long takes the website plays most:
`idle_long` (14 s: two glances, a nod, a tilt, smiles that come and go), `listen_long`
(9 s of nods and a held tilt) and `talk_long` (8 s of speech with emphasis and a glance
down), with blinks every two to four seconds. The website measures each talking clip's
mouth when it imports a run, for lip sync.

## Original music

The website's Motion videos and Aiva can use original instrumentals made here with
ACE-Step 1.5 (MIT; its makers allow commercial use of what it generates): hip-hop, R&B,
neo-soul, jazz, blues, classical, afrobeats, amapiano, lo-fi, pop and gospel, six tracks
each, listed with their sound and tempo in `music.json`. Being original, they carry no
licence fees and draw no copyright claims when posted. Run
`{"step": "music", "only": ["amapiano"], "count": 6}` to make a genre (`studio/music.py`;
one job per genre, on the CPU, written to `out/music/<genre>/` as MP3s levelled to
-14 LUFS with a JSON of what each was made from). `seconds` sets the length.

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
