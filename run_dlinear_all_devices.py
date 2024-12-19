import os

QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99] 

SPLIT = 0.9         # train/test %

FIGSIZE = (9, 6)


qL1, qL2 = 0.01, 0.10        # percentiles of predictions: lower bounds
qU1, qU2 = 1-qL1, 1-qL2,     # upper bounds derived from lower bounds
label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import missingno as mno
import pywt
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import random
import numpy as np
import torch
from tqdm import tqdm
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel, NBEATSModel, NHiTSModel, XGBModel, RNNModel, BlockRNNModel, TFTModel, TCNModel, DLinearModel, NLinearModel
from darts.models import AutoARIMA, LinearRegressionModel, RegressionEnsembleModel
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.metrics import mape, rmse, smape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner import lr_finder
import pytorch_lightning as pl
pl.seed_everything(42, workers=True)
pd.set_option("display.precision",2)
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format





xls = pd.ExcelFile('df_radon_combined.xlsx')

df_radon = {}
for num in xls.sheet_names[2:]:
    df_radon[num] = pd.read_excel(xls, num) 

#remove these indexes
devices = ['1', '2', '5', '9', '17', '23', '27', '28', '36', '44', '48']
for key in list(df_radon):
    if key in devices:
        del df_radon[key]

for key in df_radon:
    print(key)
    df_radon[key]['SyncDate'] = pd.to_datetime(df_radon[key]['SyncDate'])
    print(f"{df_radon[key]['SyncDate'].min()} - {df_radon[key]['SyncDate'].max()}")
    df_radon[key].sort_values(by='SyncDate', ascending=True, inplace=True)
    df_radon[key]['SyncDate'] = df_radon[key]['SyncDate'].dt.floor('H')
    df_radon[key] = df_radon[key].resample('H', on = 'SyncDate').mean()
    df_radon[key] = df_radon[key].interpolate(method='linear', limit_direction='both')
    start_time = pd.to_datetime('2022-05-11T18:29:00.000000000')
    end_time = pd.to_datetime('2023-06-06T12:00:00.000000000')
    df_radon[key] = df_radon[key][(df_radon[key].index >= start_time) & (df_radon[key].index <= end_time)]
    print(f"{df_radon[key].index.min()} - {df_radon[key].index.max()}")

wavelet_file = pd.read_csv('wavelet_info.csv')
#convert a column to string
wavelet_file['Device Name'] = wavelet_file['Device Name'].astype(str)
df_smape = pd.DataFrame(columns=['Device', 'sMAPE', 'in_len', 'out_len', 'kernel_size', 'const_init', 'lr', 'batch_size'])

for key in tqdm(df_radon):
    print(key)
    df = df_radon[key].copy()




    # Denoising the Radon signal
    def madev(d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)




    def wavelet_denoising(x, wavelet='db4', level=5):
        coeff = pywt.wavedec(x, wavelet, mode="per")
        n = len(x) 
        sigma = (1/0.6745) * madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        if len(x) % 2 ==0:
            return pywt.waverec(coeff, wavelet, mode='per')
        else:
            return pywt.waverec(coeff, wavelet, mode='per')[:n]





    signal = df['Radon'].copy()
    print("Wavelet Name",wavelet_file.loc[wavelet_file['Device Name'] == key, 'Wavelet'].values[0])
    filtered = wavelet_denoising(signal, wavelet=wavelet_file.loc[wavelet_file['Device Name'] == key, 'Wavelet'].values[0], level=4)
    df['Radon'] = filtered
    df['Radon_unfiltered'] = signal


    # create time series object for target variable
    ts_P = TimeSeries.from_series(df['Radon'], fill_missing_dates=True, freq="H") 
    ts_P_unfiltered = TimeSeries.from_series(df["Radon_unfiltered"], fill_missing_dates=True, freq="H")
    # check attributes of the time series
    print("components:", ts_P.components)
    print("duration:",ts_P.duration)
    print("frequency:",ts_P.freq)
    print("frequency:",ts_P.freq_str)
    print("has date time index? (or else, it must have an integer index):",ts_P.has_datetime_index)
    print("deterministic:",ts_P.is_deterministic)
    print("univariate:",ts_P.is_univariate)

    


    # train/test split and scaling of target variable
    ts_train, ts_test = ts_P.split_after(split_point=9210)
    ts_train_unfiltered, ts_test_unfiltered = ts_P_unfiltered.split_after(split_point=9210)
    print("training start:", ts_train.start_time())
    print("training end:", ts_train.end_time())
    print("training duration:",ts_train.duration)
    print("test start:", ts_test.start_time())
    print("test end:", ts_test.end_time())
    print("test duration:", ts_test.duration)


    scalerP = Scaler()
    scalerP.fit_transform(ts_train)
    ts_ttrain = scalerP.transform(ts_train)
    ts_ttest = scalerP.transform(ts_test)    
    ts_t = scalerP.transform(ts_P)

    # make sure data are of type float
    ts_t = ts_t.astype(np.float32)
    ts_ttrain = ts_ttrain.astype(np.float32)
    ts_ttest = ts_ttest.astype(np.float32)

    print("first and last row of scaled Radon time series:")
    pd.options.display.float_format = '{:,.2f}'.format
    ts_t.pd_dataframe().iloc[[0,-1]]



    print("first and last row of scaled target variable in training set: price:")
    ts_ttrain.pd_dataframe().iloc[[0,-1]]





    def set_seed(seed_value):
        import random
        import numpy as np
        import torch

        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pl.seed_everything(seed_value, workers=True)




    import torch
    from ray.air import session
    from darts.utils.losses import SmapeLoss
    from torchmetrics import MetricCollection, SymmetricMeanAbsolutePercentageError, MeanAbsolutePercentageError
    def build_fit_dlinear_model(
        model_args,
        save_checkpoints=False,
        callbacks=None,
        save_model=False
    ):
    #     BATCH_SIZE=64
        MAX_EPOCHS=500
        NR_EPOCHS_VAL_PERIOD=1
        set_seed(42)
        torch_metrics = MetricCollection([MeanAbsolutePercentageError(), SymmetricMeanAbsolutePercentageError()])

        pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices":-1,
                "auto_select_gpus": True,
                "callbacks": callbacks,
                "enable_progress_bar": False,
            }
        
    

        model = DLinearModel(
            input_chunk_length=model_args['in_len'],
            output_chunk_length=model_args['out_len'],
            batch_size=model_args['batch_size'],
            n_epochs=MAX_EPOCHS,
            nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
            model_name="DLinear",
            shared_weights=False,
            kernel_size=model_args['kernel_size'],
            const_init=model_args['const_init'],
            use_static_covariates=False,
            loss_fn=SmapeLoss(),
            optimizer_kwargs={'lr': model_args['lr']},
            log_tensorboard=False,
            force_reset=True,
            save_checkpoints=save_checkpoints,
            pl_trainer_kwargs=pl_trainer_kwargs,
            torch_metrics=torch_metrics,
            random_state=42
            )
        val_len = len(ts_test)
        val_series = ts_ttrain[-((val_len) + model_args['in_len']) :]
        ts_ttrain_input = ts_ttrain[:-(val_len )]
        model.fit(  ts_ttrain_input, 
                
                    val_series=val_series,
                    )
    #     model.load_from_checkpoint(f"{model_args['model']} RNN model", best=True)
        ts_tpred = model.predict(
                    series = ts_ttrain,
                    n = len(ts_ttest),
                    verbose=True
        )
        ts_q = scalerP.inverse_transform(ts_tpred)
        q_smape = smape(ts_q, ts_test)
        session.report({'q_smape': q_smape})





    def build_fit_dlinear_model_return(
        model_args,
        save_checkpoints=False,
        callbacks=None,
        save_model=False
    ):
    #     BATCH_SIZE=64
        MAX_EPOCHS=500
        NR_EPOCHS_VAL_PERIOD=1
        set_seed(42)
        torch_metrics = MetricCollection([MeanAbsolutePercentageError(), SymmetricMeanAbsolutePercentageError()])

    
        pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices":1,
                "auto_select_gpus": True,
                "callbacks": callbacks,
                "enable_progress_bar": True,
            }
    
    

        model = DLinearModel(
            input_chunk_length=model_args['in_len'],
            output_chunk_length=model_args['out_len'],
            batch_size=model_args['batch_size'],
            n_epochs=MAX_EPOCHS,
            nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
            model_name="DLinear",
            shared_weights=False,
            kernel_size=model_args['kernel_size'],
            const_init=model_args['const_init'],
            use_static_covariates=False,
            loss_fn=SmapeLoss(),
            optimizer_kwargs={'lr': model_args['lr']},
            log_tensorboard=False,
            force_reset=True,
            save_checkpoints=save_checkpoints,
            pl_trainer_kwargs=pl_trainer_kwargs,
            torch_metrics=torch_metrics,
            random_state=42
            )
        val_len = len(ts_test)
        val_series = ts_ttrain[-((val_len) + model_args['in_len']) :]
        ts_ttrain_input = ts_ttrain[:-(val_len )]
        model.fit(  ts_ttrain_input, 
                    
                    val_series=val_series,
                    )
    
        return model





    from ray import tune
    from ray.tune import CLIReporter
    # from ray.tune.integration.pytorch_lightning import TuneReportCallback
    from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search import ConcurrencyLimiter
    # tune_callback = TuneReportCallback(
    #     {
    #         "loss":"val_loss",
    #         "sMAPE": "val_SymmetricMeanAbsolutePercentageError",
    #     },
    #     on="validation_end",
    # )

    early_stopper = EarlyStopping(
            monitor="val_SymmetricMeanAbsolutePercentageError",
            patience=3,
            mode='min',
        )

    #define the hyperparameter search space
    config = {
        "in_len": tune.randint(8,168),#setting 168 is not a good option here as convolutions take time reducing this to 80
        "out_len":tune.randint(1,24),
        "kernel_size": tune.randint(10,50),
        "const_init":tune.choice([True, False]),
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size":tune.choice([16,32,64,128,256]),
        
    }

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["q_smape"])
    resources_per_trial = {"cpu": 5, "gpu": 0.4}

    num_samples = 100

    algo = OptunaSearch()

    algo = ConcurrencyLimiter(algo, max_concurrent=10)

    scheduler = AsyncHyperBandScheduler(max_t=100, grace_period=10, reduction_factor=2)

    train_fn_with_parameters = tune.with_parameters(build_fit_dlinear_model, callbacks=[early_stopper])

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="q_smape",
        mode="min",
        config=config,
        num_samples=num_samples,
        search_alg=algo,
        scheduler = scheduler,
        progress_reporter=reporter,
        name="dlinear_tune",
        raise_on_failed_trial=False
    )

    print("Best hyperparameters found were: ", analysis.best_config)




    early_stopper = EarlyStopping(
            monitor="val_SymmetricMeanAbsolutePercentageError",
            patience=3,
            mode='min',
        )
    best_model = build_fit_dlinear_model_return(analysis.best_config, callbacks=[early_stopper])




    ts_tpred = best_model.predict(
                    series = ts_ttrain,
                    n = len(ts_ttest),
                    verbose=True
        )




    dfY = pd.DataFrame()
    dfY['Actual Filtered'] = TimeSeries.pd_series(ts_test)
    dfY['Actual'] = TimeSeries.pd_series(ts_test_unfiltered)
    def pred(ts_tpred, ts_test):
        ts_tpred = scalerP.inverse_transform(ts_tpred)
        s = TimeSeries.pd_series(ts_tpred)
        header = "Predicted"
        dfY[header] = s
        q_smape = smape(ts_tpred, ts_test)
        print('SMAPE:',q_smape)
        return q_smape
    q_smape = pred(ts_tpred, ts_test)




    # plot the forecast
    plt.figure(100, figsize=(20, 7))
    plt.plot(dfY.index, dfY['Predicted'], color='r', label='Predicted Radon Concentration')
    plt.plot(dfY.index, dfY['Actual Filtered'], color='c', label='Denoised Radon Concentration')
    plt.plot(dfY.index, dfY['Actual'], color='b', label='Actual Radon Concentration')
    plt.legend()
    #plt.title('Radon Prediction for test set')
    plt.xlabel('Day')
    plt.ylabel('Radon Concentration(pCi/L)')
    plt.savefig(f"dlinear_{key}.png")
    # clear the plot
    plt.clf()
    # append to results to df_smape
    df_smape = df_smape.append({'Device': key, 
                                'SMAPE': q_smape, 
                                'in_len': analysis.best_config['in_len'], 
                                'out_len': analysis.best_config['out_len'],
                                'kernel_size': analysis.best_config['kernel_size'],
                                'const_init': analysis.best_config['const_init'],
                                'lr': analysis.best_config['lr'],
                                'batch_size': analysis.best_config['batch_size'],}
                                , ignore_index=True)

# save df_smape
df_smape.to_csv('dlinear_smape.csv', index=False)



