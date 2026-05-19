# Skill: `batch.msg` is required even when TGN memory is off

`pidsmaker/model.py` `embed()` at L73 unconditionally passes
`msg=batch.msg` to the encoder, regardless of whether the encoder
will consume it. The kwarg access itself raises `AttributeError` if
`batch.msg` is missing.

Implications for memory optimization:
- Do **not** `del g.msg` on per-graph `CollatableTemporalData` after
  `extract_msg_from_data`, even if `cfg.training.encoder.tgn.use_memory`
  is False. Some encoders ignore msg but the kwarg pass requires the
  attribute to exist.
- Only the global `full_data.msg` (used inside `compute_tgn_graphs`)
  can be safely skipped/sliced; that one is gated on the actual TGN
  batching method, not the per-graph msg.

Originally hit on CLEARSCOPE_E3 under velox (deep_ensemble experiment):
```
File "pidsmaker/model.py", line 73, in embed
    msg=batch.msg,
AttributeError: 'GlobalStorage' object has no attribute 'msg'
```
