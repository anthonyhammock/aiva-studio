# Motion recordings

Drop a motion file from the website's **Admin → Aiva Motion** page here
(`aiva-motion-YYYY-MM-DD.json`). Committing one on its own starts the Studio
workflow, which re-animates every persona with that person's movement (about
four hours on free runners) and commits the clips to `out/people/`.

The file holds only numbers — head angles, blinks, mouth and smile, 25 times a
second — never video or pictures of the person. See "Real motion" in the main
README.
