# Project Notes

## Sampling / temperature settings (recorded 2026-07-22)

During pipeline testing we found that the sampling temperature is **not
controllable** the way the original code assumed. Both the code and the run
metadata originally implied a fixed `temperature = 0.0` across all models; that
is not achievable with the current models.

**Claude models under test (Sonnet 5, Opus 4.8):**
- The `temperature` parameter (and `top_p` / `top_k`) was removed from the
  Anthropic API for the Claude 5 family. Sending any value returns HTTP 400
  (`temperature is deprecated for this model`), confirmed empirically against
  both `claude-sonnet-5` and `claude-opus-4-8`.
- We therefore send no temperature. Each model runs at an unspecified,
  non-configurable internal sampling setting that Anthropic does not expose as a
  number. It is **not** "reset to 1" — there simply is no knob and no published
  value. Outputs are not guaranteed deterministic.
- Code change: `src/run_models.py` `_call_anthropic` no longer passes
  `temperature`.

**Judge model (GPT-5.5, exact string `gpt-5.5-2026-04-23`):**
- GPT-5.5 only accepts the default temperature (1); `temperature=0` returns
  HTTP 400 (`does not support 0 with this model. Only the default (1) value is
  supported`). So the judge runs at temperature 1, fixed — we cannot lower it.
- Code change: `src/judge.py` `score_response` no longer passes `temperature`.

**Implications for the write-up:**
- Do not describe the runs as "temperature 0" / deterministic. The judge is
  pinned at GPT-5.5's default (1); the Claude models run at an unspecified
  internal setting. Neither is deterministic.
- Judge determinism is limited; `response_format=json_object` still constrains
  the output shape but not the sampling. Consider this when interpreting any
  judge-vs-expert agreement (Cohen's kappa) — re-scoring could vary slightly.
