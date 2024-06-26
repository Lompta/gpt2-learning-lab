## GPT2-Lab

This is a personal project where I implemented GPT-2 style inference. I wanted to take the publicly available weights from a transformer, and write the code necessary to do inference with those weights.

main.py is the implementation I ended up with, which works. draft.py is an earlier effort that did a lot of standard autoregressive transformers things, but didn't leverage the libraries to be efficient or do things in precisely the GPT-2 matching ways. Since I'm using trained weights, the match needs to be exact. scraps.py is functions I wrote that are part of the process academically, but that are done more efficiently by leveraging tools.