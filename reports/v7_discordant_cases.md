# V7 Discordant Cases

## 两侧 wins

| Setting | Candidate correct / Class wrong | Candidate wrong / Class correct | Total |
|---|---:|---:|---:|
| balanced336_e1 | 10 | 5 | 15 |
| balanced336_e4 | 9 | 8 | 17 |
| equiv_all_e1 | 11 | 20 | 31 |

JSONL 同时保存 Candidate win 与 Class win，不筛掉任何一侧。逐 case 包含 query、schema、Gold、Equivalent、Teacher 概率/entropy、S0 输出/confidence、两模型输出和 error type。

## 前 40 个 case 索引

| Setting | Direction | ID | A/B/C | S0 class | S0 conf | Teacher entropy |
|---|---|---|---|---|---:|---:|
| balanced336_e1 | candidate_wrong_class_correct | xlam_33800 | C | correct | 0.6510 | 0.8952 |
| balanced336_e1 | candidate_wrong_class_correct | xlam_3437 | A | correct | 0.5359 | 0.9552 |
| balanced336_e1 | candidate_wrong_class_correct | xlam_6150 | A | correct | 0.6014 | 0.9820 |
| balanced336_e1 | candidate_correct_class_wrong | toolace_8129 | A | correct | 0.6442 | 0.9953 |
| balanced336_e1 | candidate_correct_class_wrong | xlam_25239 | A | correct | 0.6159 | 0.9825 |
| balanced336_e1 | candidate_correct_class_wrong | xlam_6197 | B | wrong_arguments | 0.5793 | 0.8855 |
| balanced336_e1 | candidate_correct_class_wrong | xlam_12826 | C | correct | 0.6227 | 0.8100 |
| balanced336_e1 | candidate_correct_class_wrong | xlam_22510 | C | correct | 0.5910 | 0.9647 |
| balanced336_e1 | candidate_correct_class_wrong | xlam_31984 | A | correct | 0.6101 | 0.9643 |
| balanced336_e1 | candidate_wrong_class_correct | toolace_9134 | C | correct | 0.6226 | 0.8369 |
| balanced336_e1 | candidate_correct_class_wrong | xlam_33361 | A | correct | 0.6047 | 0.9915 |
| balanced336_e1 | candidate_correct_class_wrong | xlam_14262 | C | correct | 0.5880 | 0.9493 |
| balanced336_e1 | candidate_correct_class_wrong | xlam_20968 | C | correct | 0.6395 | 0.8371 |
| balanced336_e1 | candidate_wrong_class_correct | xlam_60 | C | correct | 0.6045 | 0.9614 |
| balanced336_e1 | candidate_correct_class_wrong | toolace_7220 | A | correct | 0.5738 | 0.9560 |
| balanced336_e4 | candidate_correct_class_wrong | toolace_2573 | A | correct | 0.6086 | 0.9213 |
| balanced336_e4 | candidate_wrong_class_correct | xlam_44177 | A | wrong_arguments | 0.6006 | 0.9646 |
| balanced336_e4 | candidate_correct_class_wrong | toolace_8539 | A | correct | 0.6054 | 0.9675 |
| balanced336_e4 | candidate_wrong_class_correct | xlam_6150 | A | correct | 0.6014 | 0.9820 |
| balanced336_e4 | candidate_wrong_class_correct | xlam_30461 | A | correct | 0.5935 | 0.9980 |
| balanced336_e4 | candidate_wrong_class_correct | xlam_22773 | A | correct | 0.5507 | 0.9840 |
| balanced336_e4 | candidate_correct_class_wrong | toolace_9063 | A | wrong_arguments | 0.6034 | 0.9834 |
| balanced336_e4 | candidate_correct_class_wrong | xlam_19251 | C | wrong_arguments | 0.5172 | 0.9493 |
| balanced336_e4 | candidate_correct_class_wrong | xlam_6197 | B | wrong_arguments | 0.5793 | 0.8855 |
| balanced336_e4 | candidate_wrong_class_correct | xlam_40475 | A | wrong_arguments | 0.5909 | 0.9614 |
| balanced336_e4 | candidate_correct_class_wrong | toolace_3817 | A | correct | 0.6146 | 0.9780 |
| balanced336_e4 | candidate_correct_class_wrong | xlam_31984 | A | correct | 0.6101 | 0.9643 |
| balanced336_e4 | candidate_correct_class_wrong | toolace_9134 | C | correct | 0.6226 | 0.8369 |
| balanced336_e4 | candidate_correct_class_wrong | xlam_20968 | C | correct | 0.6395 | 0.8371 |
| balanced336_e4 | candidate_wrong_class_correct | xlam_20961 | A | correct | 0.5882 | 0.9907 |
| balanced336_e4 | candidate_wrong_class_correct | xlam_11954 | C | correct | 0.6270 | 0.9831 |
| balanced336_e4 | candidate_wrong_class_correct | xlam_30887 | C | correct | 0.6073 | 0.7950 |
| equiv_all_e1 | candidate_correct_class_wrong | toolace_8792 | C | wrong_function | 0.6223 | 0.9806 |
| equiv_all_e1 | candidate_wrong_class_correct | toolace_2573 | A | correct | 0.6086 | 0.9213 |
| equiv_all_e1 | candidate_wrong_class_correct | toolace_1345 | A | correct | 0.6029 | 0.9823 |
| equiv_all_e1 | candidate_wrong_class_correct | xlam_22738 | B | correct | 0.5875 | 0.9912 |
| equiv_all_e1 | candidate_wrong_class_correct | xlam_28263 | A | correct | 0.6362 | 0.9977 |
| equiv_all_e1 | candidate_wrong_class_correct | xlam_3334 | A | correct | 0.6124 | 0.9994 |
| equiv_all_e1 | candidate_wrong_class_correct | xlam_22158 | A | correct | 0.6610 | 0.9638 |
| equiv_all_e1 | candidate_wrong_class_correct | toolace_8156 | A | wrong_arguments | 0.6215 | 0.9813 |

## 资产

- Cases: [v7_discordant_cases.jsonl](v7_discordant_cases.jsonl)
- Rows: 63
- SHA256: 294fd5094abea90d333466b8a1040240557d91d116668c719f02b52f31d5b15b
- Prediction hashes: {"balanced336_e1": {"candidate": "94c11f60bcbb0e555c5c8f1baa17779a21b70236aad4350f6eb9a3175d9d1f65", "class": "98255f22dbb1e057a7cf518a212fb7f56bcddf8b766f74edd97576c6a55ad6c2"}, "balanced336_e4": {"candidate": "57bc51b451100044e742212085b703954195216a2b5eabf10141ccc56dd1e402", "class": "59bf7078ac8c43eaa691ba19650ce8dd6ef4de14f3cc2da2939822cf7771bcc8"}, "equiv_all_e1": {"candidate": "c3aab832543464f355cd878ea006d3a7006d11ec757929870a2414dcf91f085c", "class": "d847dcbb8e356b0933db482d002a49744d3f62d51d8dc6a49f82f7bceb1ca082"}}
