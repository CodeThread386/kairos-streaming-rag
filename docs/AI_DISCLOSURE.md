# AI usage disclosure

The signed hackathon form is [`LangAI3.0_AI_Disclosure_VITV_Kairos.docx`](LangAI3.0_AI_Disclosure_VITV_Kairos.docx).
This page is the same information in the repository.

**Did the team use AI in developing this project?** Yes.

**Tool:** Claude (Anthropic), used as an agentic coding and writing assistant in a sandboxed environment where it could
write files, run commands, run the tests and run the benchmark. No AI service is called by Kairos at run time in its
default configuration.

| Area | How AI was used |
|---|---|
| Idea generation | Picking the architecture (streaming controller, decomposition, hybrid retrieval, versioned refinement) against the Theme 4 guide and its evaluation gates |
| Code | The engine, API, console, benchmark harness and tests were written by Claude, then executed and debugged in its sandbox |
| UI / UX | The live console layout and visual design |
| Content | The fictional "Tessera Labs" demo corpus, the benchmark scenarios and gold labels, the README and docs, the presentation text, and the demo-video narration script |
| Data analysis | Running the benchmark, error analysis on dev and test-1, and writing up ablations and edge cases |
| Testing / debugging | Writing and running the pytest suite and the clean-environment reproduction |
| Other | Demo-video voice: Kokoro-82M (open-source TTS) via sherpa-onnx. Run-time embeddings: WordLlama (a pre-trained, non-generative static embedding model). |

## Feature origin

| Feature | Origin | Notes |
|---|---|---|
| Theme choice, requirements and submission decisions | Self | The team chose Theme 4, set the goal (a winning, fully reproducible submission), and owns review and submission |
| Streaming retrieval controller, decomposer, hybrid retrieval, session refinement, grounded composer | AI-generated (team-directed) | Generated with Claude from the Theme 4 guide; behaviour verified with 16 tests and the replay benchmark |
| Live demo console | AI-generated (team-directed) | Vanilla HTML/JS; checked with headless-browser screenshots |
| Tessera demo corpus and benchmark scenarios | AI-generated | Fictional company, marked as synthetic; gold labels written before the engine was run on each split |
| 18F handbook corpus | Third-party, public domain | CC0, copied verbatim with its licence file |
| Benchmark harness, reports, docs, deck, video | AI-generated (team-directed) | Every number in the docs and deck comes from `python -m bench.run_bench` |

## Compliance

* No proprietary or confidential data was used. The demo corpus is fictional, and the 18F handbook is public domain.
* No API keys or credentials are in the repository.
* Third-party components are used under their licences: WordLlama (MIT), Kokoro-82M (Apache-2.0), FastAPI (MIT) and others.
