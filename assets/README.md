# Assets

From the repository root, regenerate the banner:

```sh
uv run python assets/generate.py assets/banner/blocks.txt assets/banner/binaml-banner.svg --width 1600 --height 500 --baseline 336 --title Binaml --description "The word Binaml rendered in a white, italic terminal-inspired pixel typeface on a black background."
```

Regenerate the logo:

```sh
uv run python assets/generate.py assets/logo/blocks.txt assets/logo/binaml-logo.svg --width 512 --height 512 --block 31 --grid 26 --baseline 350 --title Binaml --description "The letter B from the Binaml wordmark, accompanied by the pixel dots from the Binaml wordmark, rendered in a white, italic terminal-inspired pixel typeface on a black background."
```
