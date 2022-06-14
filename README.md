# Layout-based Causal Inference for Object Navigation

## Setup
- Clone the repository and move into the top-level directory `cd Layout-based-sTDE`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate ng`
- We provide pre-trained model of our Layout-based sTDE model in the .`trained_models` directory.
- Our settings of dataset follows previous works, please refer to [HOZ](https://github.com/sx-zhang/HOZ.git) for AI2THOR and [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation.git) for Gibson.  
## Training and Evaluation
### Train our Layout-based model 
```shell
python main.py \
      --title layoutmodel \
      --model LayoutModel \
      --workers 12 \
      --gpu-ids 0
```
### Evaluate our model without sTDE (only Layout-based model) 
```shell
python full_eval.py \
      --title layoutmodel \
      --model LayoutModel \
      --results-json layoutmodel.json \
      --gpu-ids 0
```
### Evaluate our model with sTDE (our Layout-based sTDE model) 
```shell
python full_eval.py \
        --title layoutmodel \
        --model LayoutModel \
        --results-json layoutmodel_sTDE.json \
        --gpu-ids 0 \
        --TDE_self True
```
