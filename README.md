---
title: PortfolioRAG
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# PortfolioRAG

## Hugging Face deployment

The Gradio Space serves on `0.0.0.0:7860`. On Spaces, the app always uses
port 7860; locally, `GRADIO_SERVER_PORT` can override the default.
SSR is disabled. `share=True` is not required for the Space's public URL.
On free accounts where CPU Basic is unavailable for compute Spaces, keep the
Space on ZeroGPU; the Gradio chat handler is decorated with `@spaces.GPU` so
the ZeroGPU runtime accepts the app.

Set `GOOGLE_API_KEY` as a secret in the Space settings. The UI can start
without it, but indexing and answers require a valid key.

Provide the resume PDFs in `data/` in the deployed repository. This directory
allows PDF files to be tracked, while other local data files remain ignored.
Only commit PDFs that are appropriate for a public repository.
Files uploaded directly to the Space can be removed by the workflow's force
push; keep deployment data in the source repository if using this sync.

The GitHub workflow installs dependencies and checks that the actual app serves
the UI and chatbot configuration on port 7860 before syncing. Run the same
check locally with `python scripts/check_startup.py` after installing
`requirements.txt`. It does not use a Gemini key or test generated answers.

After pushing to `main`, the Space startup logs should show
`http://0.0.0.0:7860`. The public Space is
https://huggingface.co/spaces/attem03/PortfolioRAG.
