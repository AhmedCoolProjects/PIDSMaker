## Velox

### Clearscope E3

```bash
./run.sh velox CLEARSCOPE_E3 --training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=64 --training.node_out_dim=64 --training.num_epochs=12 --featurization.emb_dim=128 --construction.time_window_size=1.0 --project=P_Mine --exp=v_cs3_b
```


### CADETS E3

```bash
./run.sh velox CADETS_E3 --training.encoder.dropout=0.3 --training.lr=0.0001 --training.node_hid_dim=256 --training.node_out_dim=256 --training.num_epochs=12 --featurization.emb_dim=256 --project=P_Mine --exp=v_ca3_b
```