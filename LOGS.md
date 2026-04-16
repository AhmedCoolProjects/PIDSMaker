```bash
python ./pidsmaker/main.py argus CLEARSCOPE_E3 --database_host localhost --force_restart=training --wandb
```

```txt
# ------ WITH MLP- ------

Task done: /srv/lustre01/project/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/forks/PIDSMakerOrgFork/artifacts/triage/triage/5c1db12e77eae42239082179697c0326f98b7b3729cec28f75b2d79ed91aafec/CLEARSCOPE_E3

2026-04-15 13:12:15 - ============================================================
2026-04-15 13:12:15 - Run finished.
2026-04-15 13:12:15 - ============================================================
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:                          accuracy █▇▇▅▇▆▇▅▁▅▂▃▅▅▄▅▆▆▅▆▃▆▆▇▆▇▅
wandb:                         adp_score ▅▁▁▁▁▂▄▅▅█▂▄▃▄▂▅▅▄▄▄▄▅▄▇▂▅█
wandb:                                ap ▇▂▁▁▁▂▂▃▄█▂▅▃▅▃▇▇▅▃▃▃▅▃▄▂▃█
wandb:                               auc █▅▂▁▁▂▃▅▆▆▃▆▄▆▅▆▆▆▂▆▃▅▂▅▂▃▆
wandb:                      balanced_acc ▁▁▁▁▁▂▂▂▇█▄█▄█▄██▄▄▃▃▄▄▄▂▄█
wandb:               discrim_score_att_0 ▁▆▇▇█▆▅▅▇▆▄▇▅▇▅▇▇▆▅▇▆█▇█▇█▆
wandb:                  discrim_tp_att_0 ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                discrim_tp_att_sum ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                    discrimination ▁▆▇▇█▆▅▅▇▆▄▇▅▇▅▇▇▆▅▇▆█▇█▇█▆
wandb:                               dor ▁▁▁▁▁▂▂▁▃▆▂▄▃▆▃▆█▄▃▃▂▄▄▅▂▄▆
wandb:                             epoch ▁▁▁▂▂▃▃▃▄▄▅▅▅▆▆▇▇▇██▁▁▂▂▃▃▃▄▄▄▅▅▅▆▆▇▇▇█▃
wandb:                                fn █████▇▇▇▁▂▅▁▅▁▅▂▁▅▅▆▅▅▅▅▇▅▂
wandb:                                fp ▁▂▂▄▂▃▂▄█▄▇▆▄▄▅▄▃▃▄▃▆▃▃▂▃▂▄
wandb:                               fpr ▁▂▂▄▂▃▂▄█▄▇▆▄▄▅▄▃▃▄▃▆▃▃▂▃▂▄
wandb:       fps_if_all_attacks_detected ▃▂▁▁▂▃▁▁▁▁▆▁▂▁▃▂▂▂▂▁█▁▂▁▃▁▁
wandb:                            fscore ▁▁▁▁▁▂▂▂▄▆▃▅▄▆▃▇█▄▄▄▃▄▄▅▂▅▆
wandb:                             lr(+) ▁▁▁▁▁▂▃▂▃▆▂▄▄▆▃▆█▄▃▄▂▄▄▆▂▅▆
wandb:                               mcc ▁▁▁▁  ▂ ▅   ▃   ▇▅▃ ▂ █    
wandb:         peak_inference_cpu_memory ▁
wandb:         peak_inference_gpu_memory ▁
wandb:             peak_train_cpu_memory ▁
wandb:             peak_train_gpu_memory ▁
wandb:                         precision ▁▁▁▁▁▂▃▂▃▆▂▄▄▆▃▆█▄▄▄▂▄▄▆▂▅▆
wandb: precision_if_all_attacks_detected ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                            recall ▁▁▁▁▁▂▂▂█▇▄█▄█▄▇█▄▄▃▄▄▄▄▂▄▇
wandb:    recall_if_all_attacks_detected ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                         test_loss █▅▃▃▂▂▂▂▂▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                     time_batching ▁
wandb:                 time_construction ▁
wandb:                   time_evaluation ▁
wandb:               time_feat_inference ▁
wandb:                time_featurization ▁
wandb:          time_per_batch_inference ▁
wandb:                     time_training ▁
wandb:               time_transformation ▁
wandb:                       time_triage ▁
wandb:                                tn █▇▇▅▇▆▇▅▁▅▂▃▅▅▄▅▆▆▅▆▃▆▆▇▆▇▅
wandb:                                tp ▁▁▁▁▁▂▂▂█▇▄█▄█▄▇█▄▄▃▄▄▄▄▂▄▇
wandb:       tps_if_all_attacks_detected ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                       train_epoch ▁▁▁▂▂▂▃▃▃▃▄▄▄▅▅▅▅▆▆▆▇▇▇▇██
wandb:                  train_epoch_time ▁
wandb:                        train_loss █▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                          val_loss █▄▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                         val_score ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  
wandb: 
wandb: Run summary:
wandb:                          accuracy 0.97005
wandb:                         adp_score 0.129
wandb:                                ap 0.02527
wandb:                               auc 0.66723
wandb:                      balanced_acc 0.62942
wandb:               discrim_score_att_0 -0.3155
wandb:                  discrim_tp_att_0 0
wandb:                discrim_tp_att_sum 0
wandb:                    discrimination -0.3155
wandb:                               dor 14.61137
wandb:                             epoch 17
wandb:                                fn 2388
wandb:                                fp 18246
wandb:                               fpr 0.02661
wandb:       fps_if_all_attacks_detected 83
wandb:                            fscore 0.08464
wandb:                             lr(+) 10.7259
wandb:                               mcc nan
wandb:              neat_scores_img_file /srv/lustre01/projec...
wandb:         peak_inference_cpu_memory 0.036
wandb:         peak_inference_gpu_memory 0.033
wandb:             peak_train_cpu_memory 0.006
wandb:             peak_train_gpu_memory 0.038
wandb:                         precision 0.04969
wandb: precision_if_all_attacks_detected 0.0
wandb:                            recall 0.28546
wandb:    recall_if_all_attacks_detected 0.0
wandb:                       scores_file /srv/lustre01/projec...
wandb:                         test_loss 0.481
wandb:                     time_batching 0.04
wandb:                 time_construction 0.04
wandb:                   time_evaluation 642.57
wandb:               time_feat_inference 0.03
wandb:                time_featurization 0.03
wandb:          time_per_batch_inference 0.014
wandb:                     time_training 945.76
wandb:               time_transformation 0.04
wandb:                       time_triage 0.05
wandb:                                tn 667336
wandb:                                tp 954
wandb:       tps_if_all_attacks_detected 0
wandb:                       train_epoch 49
wandb:                  train_epoch_time 8.66
wandb:                        train_loss 0.3653
wandb:                          val_loss 0.5166
wandb:                         val_score -inf
wandb: 
wandb: 🚀 View run CLEARSCOPE_E3_argus at: https://wandb.ai/ahmed-um6p-um6p/PIDSMaker/runs/6ti31nt5
wandb: ⭐️ View project at: https://wandb.ai/ahmed-um6p-um6p/PIDSMaker
wandb: Synced 6 W&B file(s), 78 media file(s), 2 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20260415_124541-6ti31nt5/logs


# ------ WITH GAT- ------
 2026-04-15 13:56:50 - recall: 0.00209
2026-04-15 13:56:50 - fpr: 0.0126331
2026-04-15 13:56:50 - fscore: 0.00117
2026-04-15 13:56:50 - ap: 0.01267
2026-04-15 13:56:50 - accuracy: 0.98259
2026-04-15 13:56:50 - balanced_acc: 0.49473
2026-04-15 13:56:50 - auc: 0.71012
2026-04-15 13:56:50 - lr(+): 0.1658
2026-04-15 13:56:50 - dor: 0.16405
2026-04-15 13:56:50 - mcc: nan
2026-04-15 13:56:50 - tp: 7
2026-04-15 13:56:50 - fp: 8661
2026-04-15 13:56:50 - tn: 676921
2026-04-15 13:56:50 - fn: 3335
2026-04-15 13:56:50 - fps_if_all_attacks_detected: 5697
2026-04-15 13:56:50 - tps_if_all_attacks_detected: 0
2026-04-15 13:56:50 - precision_if_all_attacks_detected: 0.0
2026-04-15 13:56:50 - recall_if_all_attacks_detected: 0.0
2026-04-15 13:56:50 - adp_score: 0.02
2026-04-15 13:56:50 - discrim_score_att_0: -0.593
2026-04-15 13:56:50 - discrimination: -0.593
2026-04-15 13:56:50 - discrim_tp_att_0: 0
2026-04-15 13:56:50 - discrim_tp_att_sum: 0
2026-04-15 13:56:50 - scores_file: /srv/lustre01/project/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/forks/PIDSMakerOrgFork/artifacts/evaluation/evaluation/df59f3b006891bb2799436f3eab457c61e4af846cf18082c6a42093909fc5b09/CLEARSCOPE_E3/precision_recall_dir/scores_model_epoch_47.pkl
2026-04-15 13:56:50 - neat_scores_img_file: /srv/lustre01/project/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/forks/PIDSMakerOrgFork/artifacts/evaluation/evaluation/df59f3b006891bb2799436f3eab457c61e4af846cf18082c6a42093909fc5b09/CLEARSCOPE_E3/precision_recall_dir/neat_scores_model_epoch_47.svg

2026-04-15 13:56:50 - [@model_epoch_49] - Test Evaluation
2026-04-15 13:56:52 - Thresholds: MEAN=0.577, STD=0.352, MAX=3.499, 90 Percentile=0.987
2026-04-15 13:56:52 - Threshold: 3.499
2026-04-15 13:56:52 - Compute edge labels...
2026-04-15 13:56:57 - Found 3342 / 9851 malicious edges
2026-04-15 13:56:57 - Saving figures to /srv/lustre01/project/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/forks/PIDSMakerOrgFork/artifacts/evaluation/evaluation/df59f3b006891bb2799436f3eab457c61e4af846cf18082c6a42093909fc5b09/CLEARSCOPE_E3/precision_recall_dir/...
/srv/lustre01/project/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/forks/PIDSMakerOrgFork/pidsmaker/detection/evaluation_methods/evaluation_utils.py:97: RuntimeWarning: overflow encountered in scalar multiply
  denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
/srv/lustre01/project/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/forks/PIDSMakerOrgFork/pidsmaker/detection/evaluation_methods/evaluation_utils.py:97: RuntimeWarning: invalid value encountered in power
  denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
2026-04-15 13:57:19 - [@model_epoch_49] - Stats
2026-04-15 13:57:19 - precision: 0.00875
2026-04-15 13:57:19 - recall: 0.02902
2026-04-15 13:57:19 - fpr: 0.016036
2026-04-15 13:57:19 - fscore: 0.01344
2026-04-15 13:57:19 - ap: 0.01068
2026-04-15 13:57:19 - accuracy: 0.97933
2026-04-15 13:57:19 - balanced_acc: 0.50649
2026-04-15 13:57:19 - auc: 0.72821
2026-04-15 13:57:19 - lr(+): 1.80996
2026-04-15 13:57:19 - dor: 1.83417
2026-04-15 13:57:19 - mcc: nan
2026-04-15 13:57:19 - tp: 97
2026-04-15 13:57:19 - fp: 10994
2026-04-15 13:57:19 - tn: 674588
2026-04-15 13:57:19 - fn: 3245

``` 


Here is the \LaTeX{} snippet for your manuscript:

\begin{quote}
\noindent\textbf{Sentence Construction.} For each node 
𝑣
v
 in our provenance graph, we select all its neighbors within a 1-hop radius and their reduced interactions, sorted temporally. We then follow a \textbf{directional serialization template} to transform the local subgraph into a semantic sequence. Each sentence begins with the anchor node 
𝑣
v
, followed by a sequence of triplets consisting of a directional tag (
⟨
𝐼
𝑁
⟩
⟨IN⟩
 or 
⟨
𝑂
𝑈
𝑇
⟩
⟨OUT⟩
), the specific system call, and the neighboring entity. This formulation explicitly encodes the node's role as either the subject or object of an action, preserving both the causal flow and the behavioral context for the subsequent embedding process.
\end{quote}

Technical Note for your writing: I used the term "directional serialization template" as it sounds more robust for a PhD-level publication. Ensure you define the 
⟨
𝐼
𝑁
⟩
⟨IN⟩
 and 
⟨
𝑂
𝑈
𝑇
⟩
⟨OUT⟩
 tags in your "Preprocessing" or "Implementation Details" section.