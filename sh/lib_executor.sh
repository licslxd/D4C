# shellcheck shell=bash
# torchrun 加载用历史文件名（仅高级/非主线覆盖；日常请 python code/d4c.py …）。
# 覆盖方式: export D4C_EXEC_ADVTRAIN=my_adv.sh 前 source 本文件（仅高级场景）。
: "${D4C_EXEC_ADVTRAIN:=AdvTrain.py}"
: "${D4C_EXEC_STEP4:=generate_counterfactual.py}"
: "${D4C_EXEC_STEP5:=run-d4c.py}"
