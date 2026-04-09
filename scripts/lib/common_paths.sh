# shellcheck shell=bash
# 路径预测：与 code/d4c_core/path_layout 一致。依赖：已设置 D4C_ROOT。

d4c_predict_step3_train_log() {
    local tid="${1:?task id}"
    (cd "${D4C_ROOT}/code" && PYTHONPATH=. D4C_ROOT_VAL="$D4C_ROOT" D4C_TID_VAL="$tid" D4C_ITER_VAL="${D4C_ITER:-v1}" D4C_RUN_ID_VAL="${D4C_RUN_ID:-}" python -c "
import os
from pathlib import Path
from d4c_core import path_layout, run_naming
root = Path(os.environ['D4C_ROOT_VAL']).resolve()
tid = int(os.environ['D4C_TID_VAL'])
iter_id = os.environ['D4C_ITER_VAL']
parent = path_layout.get_iteration_root(root, tid, iter_id) / 'train' / 'step3'
ex = os.environ.get('D4C_RUN_ID_VAL', '').strip()
if ex:
    rid = run_naming.parse_run_id(ex)
else:
    rid = run_naming.next_run_id(parent)
run_root = path_layout.get_train_step3_run_root(root, tid, iter_id, rid)
print(path_layout.logs_dir(run_root) / 'train.log')
")
}

d4c_predict_step3_eval_log() {
    local tid="${1:?task id}"
    (cd "${D4C_ROOT}/code" && PYTHONPATH=. D4C_ROOT_VAL="$D4C_ROOT" D4C_TID_VAL="$tid" D4C_ITER_VAL="${D4C_ITER:-v1}" D4C_RUN_ID_VAL="${D4C_RUN_ID:-}" python -c "
import os
from pathlib import Path
from d4c_core import path_layout, run_naming
root = Path(os.environ['D4C_ROOT_VAL']).resolve()
tid = int(os.environ['D4C_TID_VAL'])
iter_id = os.environ['D4C_ITER_VAL']
parent = path_layout.get_iteration_root(root, tid, iter_id) / 'train' / 'step3'
ex = os.environ.get('D4C_RUN_ID_VAL', '').strip()
if ex:
    rid = run_naming.parse_run_id(ex)
else:
    rid = run_naming.next_run_id(parent)
run_root = path_layout.get_train_step3_run_root(root, tid, iter_id, rid)
print(path_layout.logs_dir(run_root) / 'eval.log')
")
}

d4c_predict_step5_train_log() {
    local tid="${1:?task id}"
    (cd "${D4C_ROOT}/code" && PYTHONPATH=. D4C_ROOT_VAL="$D4C_ROOT" D4C_TID_VAL="$tid" D4C_ITER_VAL="${D4C_ITER:-v1}" D4C_RUN_ID_VAL="${D4C_STEP5_RUN_ID:-}" python -c "
import os
from pathlib import Path
from d4c_core import path_layout, run_naming
root = Path(os.environ['D4C_ROOT_VAL']).resolve()
tid = int(os.environ['D4C_TID_VAL'])
iter_id = os.environ['D4C_ITER_VAL']
parent = path_layout.get_iteration_root(root, tid, iter_id) / 'train' / 'step5'
ex = os.environ.get('D4C_RUN_ID_VAL', '').strip()
if ex:
    rid = run_naming.parse_run_id(ex)
else:
    rid = run_naming.next_run_id(parent)
run_root = path_layout.get_train_step5_run_root(root, tid, iter_id, rid)
print(path_layout.logs_dir(run_root) / 'train.log')
")
}

d4c_predict_eval_log() {
    local tid="${1:?task id}"
    (cd "${D4C_ROOT}/code" && PYTHONPATH=. D4C_ROOT_VAL="$D4C_ROOT" D4C_TID_VAL="$tid" D4C_ITER_VAL="${D4C_ITER:-v1}" D4C_RUN_ID_VAL="${D4C_EVAL_RUN_ID:-}" python -c "
import os
from pathlib import Path
from d4c_core import path_layout, run_naming
root = Path(os.environ['D4C_ROOT_VAL']).resolve()
tid = int(os.environ['D4C_TID_VAL'])
iter_id = os.environ['D4C_ITER_VAL']
parent = path_layout.get_iteration_root(root, tid, iter_id) / 'eval'
ex = os.environ.get('D4C_RUN_ID_VAL', '').strip()
if ex:
    rid = run_naming.parse_run_id(ex)
else:
    rid = run_naming.next_run_id(parent)
run_root = path_layout.get_eval_run_root(root, tid, iter_id, rid)
print(path_layout.logs_dir(run_root) / 'eval.log')
")
}
