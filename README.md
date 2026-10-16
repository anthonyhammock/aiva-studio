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

## Running it

Edit `request.json` and push it; `.github/workflows/studio.yml` does the rest and
commits results back. The fields are:

- `step`: `portraits` or `animate`
- `only`: persona ids; leave it empty for all twelve
- `count`: number of candidates per persona
- `model`: `lightning` (fast) or `full`
- `clips`: a subset of clips to (re)make

You can also start it from Actions → Studio → Run workflow.

## Credits and licences

See [NOTICE.md](NOTICE.md).
