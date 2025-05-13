# state-eval

Set config parameters in `replogle_config.yaml`

```
python -m eval.run_eval --adata_pred '/home/jeremys/code/state-eval/adata_pred_subset.h5ad' \
    --adata_true '/home/jeremys/code/state-eval/adata_true_subset.h5ad' \
    --eval_config '/home/jeremys/code/state-eval/config/replogle_config.yaml'
```

## Install



### Rust/Cargo
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
rustc --version
cargo --version
```

### Wheel
```bash
# Build wheel for distribution
pip install build
python -m build

# Install from distributed wheel file
pip install ./dist/state_eval-0.1.12-py3-none-any.whl
```

### Conda Env
```bash
conda env export > environment.yml
conda env create -f environment.yml
```