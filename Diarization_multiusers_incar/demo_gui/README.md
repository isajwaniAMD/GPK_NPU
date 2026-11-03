# Speaker-aware voice assistant: browser demo

Plays `ASR_LLM_inference/incar_commands_us_en.wav` (4 speakers, 10 questions) while the pipeline runs and shows each
answer under the question that asked it. Diarization and Parakeet use the NPU, Qwen3-4B uses the iGPU.

## Run

```bash
~/isdev/NPU_WL/parakeetnpu_multizone_diarization/demo_gui/run_demo.sh
```

It starts the LLM server (iGPU) if it is not already up, then the app. Wait for `models ready ... open http://127.0.0.1:8080`
(about 15 s), then open http://127.0.0.1:8080 in Chrome and press **Start demo**.

Over SSH, forward the port from your laptop first: `ssh -L 8080:127.0.0.1:8080 isajwani@<host>`, then browse to
http://127.0.0.1:8080 on the laptop.

## Choosing a clip

The **Audio clip** dropdown in the Demo card lists every usable clip (16 kHz mono 16-bit WAV, at most 5 minutes) found in
`ASR_LLM_inference/`, `audio_files/` and `diarization/audio/`, so you can run different use cases: clean in-car questions,
speakers talking over each other, overlapping speech with background music, a team meeting, a real recording. Pick one,
then press **Start demo**. To add your own, drop a WAV into `audio_files/` (convert with
`ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le out.wav`) and reload the page. `run_demo.sh path/to/clip.wav` sets the clip
selected when the page opens.

## Simulated answers and unanswered turns

The assistant has no real map or vehicle connection. `ASR_LLM_inference/asr_llm_pipeline.py` defines a small block of made-up
car state (`VEHICLE`: battery and range, nearest charger, charging time, cabin temperature, playlists, reminders) that goes
into the LLM prompt, so answers like "the nearest charger is 4.2 miles away" are possible. They are simulated demo values, not
real data; the page does not display them. Edit `VEHICLE` to change the scenario.

When two voices speak at the same time, Parakeet may return nothing (or cut off words) for that turn. The turn card then shows
"Couldn't make out this turn" and **NO ANSWER**; no LLM call is made. Separating overlapping voices is not implemented.

## Stop

`Ctrl+C` in the terminal running `run_demo.sh` stops the app. The LLM server keeps running in the background
(log: `demo_gui/llm_server.log`); stop it with `pkill -f llm_server.py`.

## Files

- `app.py`: FastAPI backend (runs the pipeline, streams events to the page, serves `/api/usage`)
- `index.html`: the page (no external libraries)
- `run_demo.sh`: starts everything

## Compute usage panel

The left column shows live NPU / iGPU / CPU usage (updated every 0.5 s, 30 s sparkline). CPU is system-wide from `/proc/stat`,
the iGPU is the kernel's `gpu_busy_percent` (system-wide, so it includes the LLM server), and the NPU is the share of each
sample window in which one of the demo's NPU inferences (diarizer step graph, Parakeet encoder) was running. The NPU driver
exposes no utilization counter, so that value reflects this demo's own NPU work, not other processes.
