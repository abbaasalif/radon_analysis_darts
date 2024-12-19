
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
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel, NBEATSModel, NHiTSModel, XGBModel, RNNModel, BlockRNNModel
from darts.models import AutoARIMA, LinearRegressionModel, RegressionEnsembleModel
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis
from darts.metrics import mape, rmse, smape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner import lr_finder
import pytorch_lightning as pl
from tqdm import tqdm
pl.seed_everything(42)
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
#convert a column to a string
wavelet_file['Device Name'] = wavelet_file['Device Name'].astype(str)
df_smape = df_smape = pd.DataFrame(columns=['Device', 'SMAPE', 'in_len', 'out_len', 'lr', 'batch_size'])


for key in ['13', '45']:
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
    wavelet_name='coif9'
    filtered = wavelet_denoising(signal, wavelet=wavelet_file.loc[wavelet_file['Device Name'] == key, 'Wavelet'].values[0], level=4)
    df['Radon'] = filtered
    df['Radon_unfiltered'] = signal


    weather_station = pd.read_csv('weather_data_combined.csv')


    weather_station.drop('Date', axis=1, inplace=True)
    weather_station['Simple Date'] = pd.to_datetime(weather_station['Simple Date'])
    weather_station.sort_values(by='Simple Date', ascending=True, inplace=True)
    weather_station['Simple Date'] = weather_station['Simple Date'].dt.floor('H')
    weather_station = weather_station.resample('H', on = 'Simple Date').mean()
    weather_station = weather_station.interpolate(method='linear', limit_direction='both')
    for column in weather_station.columns:
        df[column] = weather_station[column]
    df.dropna(inplace=True)
    for column in df.columns:
        df[column] = df[column].astype('float32')
        if column == 'Radon':
            continue
        else:
            for i in range(1,40):
                df[column+'_lag_'+str(i)] = df[column].shift(i)
    df.fillna(0, inplace=True)
    #finding correlations with Radon
    df_corr = df.corr(method="spearman")
    print(df_corr.shape)
    print("correlation with Radon:")
    df_corrP = pd.DataFrame(df_corr["Radon"].sort_values(ascending=False))
    df_corrP.drop(["AirPressure", "Humidity","Temperature","Outdoor h Temperature (°F)","Wind Speed (mph)","Wind Gust (mph)","Max Daily Gust (mph)","Wind Direction (°)","Hourly Rain (in/hr)","Event Rain (in)","Daily Rain (in)","Weekly Rain (in)","Monthly Rain (in)","Yearly Rain (in)","Relative Pressure (inHg)","Humidity (%)","Ultra-Violet Radiation Index","Solar Radiation (W/m^2)","Absolute Pressure (inHg)","Avg Wind Direction (10 mins) (°)","Avg Wind Speed (10 mins) (mph)"], axis=0, inplace=True)
    df_corrH = df_corrP.head(2)
    df4 = df[df_corrH.index]   # keep the components with at least modest correlations
    df4.info()
    ts_P = TimeSeries.from_series(df4["Radon"], fill_missing_dates=True, freq="H") 
    ts_P_unfiltered = TimeSeries.from_series(df["Radon_unfiltered"], fill_missing_dates=True, freq="H")
    print("components:", ts_P.components)
    print("duration:",ts_P.duration)
    print("frequency:",ts_P.freq)
    print("frequency:",ts_P.freq_str)
    print("has date time index? (or else, it must have an integer index):",ts_P.has_datetime_index)
    print("deterministic:",ts_P.is_deterministic)
    print("univariate:",ts_P.is_univariate)
    df_covF = df4.loc[:, df4.columns != "Radon"]
    df_covF = df_covF.loc[:, df_covF.columns != 'Outdoor h Temperature (°F)_lag_25']
    ts_covF = TimeSeries.from_dataframe(df_covF, fill_missing_dates=True, freq="H")
    print("components (columns) of feature time series:", ts_covF.components)
    print("duration:",ts_covF.duration)
    print("frequency:",ts_covF.freq)
    print("frequency:",ts_covF.freq_str)
    print("has date time index? (or else, it must have an integer index):",ts_covF.has_datetime_index)
    print("deterministic:",ts_covF.is_deterministic)
    print("univariate:",ts_covF.is_univariate)
    ar_covF = ts_covF.all_values()

    df_covF = ts_covF.pd_dataframe()

    # train/test split and scaling of target variable
    ts_train, ts_test = ts_P.split_after(split_point=9210)
    ts_train_unfiltered, ts_test_unfiltered = ts_P_unfiltered.split_after(split_point=9210)
    scalerP = Scaler()
    scalerP.fit_transform(ts_train)
    ts_ttrain = scalerP.transform(ts_train)
    ts_ttest = scalerP.transform(ts_test)    
    ts_t = scalerP.transform(ts_P)

    # make sure data are of type float
    ts_t = ts_t.astype(np.float32)
    ts_ttrain = ts_ttrain.astype(np.float32)
    ts_ttest = ts_ttest.astype(np.float32)



    # train/test split and scaling of feature covariates
    covF_train, covF_test = ts_covF.split_after(split_point=9210)

    scalerF = Scaler()
    scalerF.fit_transform(covF_train)
    covF_ttrain = scalerF.transform(covF_train) 
    covF_ttest = scalerF.transform(covF_test)   
    covF_t = scalerF.transform(ts_covF)  
    covF_ttrain = covF_ttrain.astype(np.float32)
    covF_ttest = covF_ttest.astype(np.float32)

    import torch
    from ray.air import session
    from darts.utils.losses import SmapeLoss
    from torchmetrics import MetricCollection, SymmetricMeanAbsolutePercentageError, MeanAbsolutePercentageError
    def build_fit_nbeats_model(
        model_args,
        save_checkpoints=False,
        callbacks=None,
        save_model=False
    ):

        MAX_EPOCHS=500
        NR_EPOCHS_VAL_PERIOD=1

        torch_metrics = MetricCollection([MeanAbsolutePercentageError(), SymmetricMeanAbsolutePercentageError()])


        pl_trainer_kwargs={
                "accelerator": "gpu",
                "gpus":-1,
                "auto_select_gpus": True,
                "callbacks": callbacks,
                "enable_progress_bar": False,
            }
        encoders={"cyclic": {"past": ["hour"]},
                'transformer':Scaler()} if model_args['include_hour'] else None
        if model_args['layer_widths'] == 'a':
            layer_widths = [64,64]
        elif model_args['layer_widths'] == 'b':
            layer_widths = [128,128]
        elif model_args['layer_widths'] == 'c':
            layer_widths = [256,256]
        elif model_args['layer_widths'] == 'd':
            layer_widths = [64,128]
        elif model_args['layer_widths'] == 'e':
            layer_widths = [128,256]
        elif model_args['layer_widths'] == 'f':
            layer_widths = [256,512]
        elif model_args['layer_widths'] == 'g':
            layer_widths = [64, 256]

        model = NBEATSModel(
            input_chunk_length=model_args['in_len'],
            output_chunk_length=model_args['out_len'],
            batch_size=model_args['batch_size'],
            n_epochs=MAX_EPOCHS,
            nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
            model_name="NBEATS",
            generic_architecture = True,
            num_stacks=2,
            num_blocks = model_args['num_blocks'],
            num_layers = model_args['num_layers'],
            layer_widths = layer_widths,
            expansion_coefficient_dim = model_args['expansion_coefficient_dim'],
            dropout=model_args['dropout'],
            activation = model_args['activation'],
            loss_fn=SmapeLoss(),
            optimizer_kwargs={'lr': model_args['lr']},
            add_encoders=encoders,
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
                    past_covariates=covF_t,
                    val_series=val_series,
                    val_past_covariates=covF_t,)

        ts_tpred = model.predict(
                    series = ts_ttrain,
                    past_covariates=covF_t,
                    n = len(ts_ttest),
                    verbose=True
        )
        ts_q = scalerP.inverse_transform(ts_tpred)
        q_smape = smape(ts_q, ts_test)
        session.report({'q_smape': q_smape})


    def build_fit_nbeats_model_return(
        model_args,
        save_checkpoints=False,
        callbacks=None,
        save_model=False
    ):

        MAX_EPOCHS=500
        NR_EPOCHS_VAL_PERIOD=1

        torch_metrics = MetricCollection([MeanAbsolutePercentageError(), SymmetricMeanAbsolutePercentageError()])


        pl_trainer_kwargs={
                "accelerator": "gpu",
                "gpus":1,
                "auto_select_gpus": True,
                "callbacks": callbacks,
                "enable_progress_bar": False,
            }
        encoders={"cyclic": {"past": ["hour"]},
                'transformer':Scaler()} if model_args['include_hour'] else None
        
        if model_args['layer_widths'] == 'a':
            layer_widths = [64,64]
        elif model_args['layer_widths'] == 'b':
            layer_widths = [128,128]
        elif model_args['layer_widths'] == 'c':
            layer_widths = [256,256]
        elif model_args['layer_widths'] == 'd':
            layer_widths = [64,128]
        elif model_args['layer_widths'] == 'e':
            layer_widths = [128,256]
        elif model_args['layer_widths'] == 'f':
            layer_widths = [256,512]
        elif model_args['layer_widths'] == 'g':
            layer_widths = [64, 256]

        model = NBEATSModel(
            input_chunk_length=model_args['in_len'],
            output_chunk_length=model_args['out_len'],
            batch_size=model_args['batch_size'],
            n_epochs=MAX_EPOCHS,
            nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
            model_name="NBEATS",
            generic_architecture = True,
            num_stacks=2,
            num_blocks = model_args['num_blocks'],
            num_layers = model_args['num_layers'],
            layer_widths =layer_widths,
            expansion_coefficient_dim = model_args['expansion_coefficient_dim'],
            dropout=model_args['dropout'],
            activation = model_args['activation'],
            loss_fn=SmapeLoss(),
            optimizer_kwargs={'lr': model_args['lr']},
            add_encoders=encoders,
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
                    past_covariates=covF_t,
                    val_series=val_series,
                    val_past_covariates=covF_t,)

        return model


    # from ray import tune
    # from ray.tune import CLIReporter
    # # from ray.tune.integration.pytorch_lightning import TuneReportCallback
    # from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler
    # from ray.tune.search.optuna import OptunaSearch
    # from ray.tune.search import ConcurrencyLimiter


    # early_stopper = EarlyStopping(
    #         monitor="val_SymmetricMeanAbsolutePercentageError",
    #         patience=3,
    #         mode='min',
    #     )

    # define the hyperparameter search space
    # config = {"in_len": tune.randint(8,30),
    #     "out_len":tune.randint(1,4),
    #     "batch_size":tune.choice([32,64,128,256]),
    #     "num_blocks": tune.randint(1,10),
    #     "num_layers":tune.randint(1,5),
    #     "layer_widths":tune.choice(['a','b','c','d','e','f','g']),
    #     "expansion_coefficient_dim":tune.randint(10, 50),
    #     "dropout":tune.uniform(0.1,0.5),
    #     "activation":tune.choice(['ReLU','RReLU', 'PReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU','Sigmoid']),
    #     "lr":tune.loguniform(1e-5,1e-1),
    #     "include_hour":tune.choice([True,False]),
         
    # }

    # reporter = CLIReporter(
    #     parameter_columns=list(config.keys()),
    #     metric_columns=["q_smape"])
    # resources_per_trial = {"cpu": 5, "gpu": 0.4}

    # num_samples = 100

    # algo = OptunaSearch()

    # algo = ConcurrencyLimiter(algo, max_concurrent=10)

    # scheduler = AsyncHyperBandScheduler(max_t=100, grace_period=10, reduction_factor=2)

    # train_fn_with_parameters = tune.with_parameters(build_fit_nbeats_model, callbacks=[early_stopper])

    # analysis = tune.run(
    #     train_fn_with_parameters,
    #     resources_per_trial=resources_per_trial,
    #     metric="q_smape",
    #     mode="min",
    #     config=config,
    #     num_samples=num_samples,
    #     search_alg=algo,
    #     scheduler = scheduler,
    #     progress_reporter=reporter,
    #     name="nbeats_tune_cov",
    #     raise_on_failed_trial=False
    # )

    # print("Best hyperparameters found were: ", analysis.best_config)

    if key == '13':
        best_config = {'in_len': 15, 'out_len': 2, 'batch_size': 128, 'num_blocks': 7, 'num_layers': 2, 'layer_widths': 'g', 'expansion_coefficient_dim': 49, 'dropout': 0.226208871931251, 'activation': 'LeakyReLU', 'lr': 0.000204263652840103, 'include_hour': True}
    elif key == '45':
        best_config = {'in_len': 23, 'out_len': 2, 'batch_size': 128, 'num_blocks': 3, 'num_layers': 2, 'layer_widths': 'b', 'expansion_coefficient_dim': 30, 'dropout': 0.20560205296728, 'activation': 'RReLU', 'lr': 0.000203942927012489, 'include_hour': False}


    early_stopper = EarlyStopping(
            monitor="val_SymmetricMeanAbsolutePercentageError",
            patience=3,
            mode='min',
        )
    best_model = build_fit_nbeats_model_return(best_config, callbacks=[early_stopper])


    ts_tpred = best_model.predict(
                    series = ts_ttrain,
                    past_covariates=covF_t,
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

    #save dfY
    dfY.to_csv(f"nbeats_{key}.csv", index=True)


    # # plot the forecast
    # plt.figure(100, figsize=(20, 7))
    # plt.plot(dfY.index, dfY['Predicted'], color='r', label='Predicted Radon Concentration')
    # plt.plot(dfY.index, dfY['Actual Filtered'], color='c', label='Denoised Radon Concentration')
    # plt.plot(dfY.index, dfY['Actual'], color='b', label='Actual Radon Concentration')
    # plt.legend()
    # #plt.title('Radon Prediction for test set')
    # plt.xlabel('Day')
    # plt.ylabel('Radon Concentration(pCi/L)')
    # plt.savefig(f"nbeats_{key}.png")
    # # clear the plot
    # plt.clf()

    # df_smape = df_smape.append({'Device': key, 
    #                             'SMAPE': q_smape, 
    #                             'in_len': analysis.best_config['in_len'], 
    #                             'out_len': analysis.best_config['out_len'],
    #                             'dropout': analysis.best_config['dropout'],
    #                             'num_blocks': analysis.best_config['num_blocks'],
    #                             'num_layers': analysis.best_config['num_layers'],
    #                             'expansion_coefficient_dim': analysis.best_config['expansion_coefficient_dim'],
    #                             'activation': analysis.best_config['activation'],
    #                             'layer_widths': analysis.best_config['layer_widths'],
    #                             'lr': analysis.best_config['lr'],
    #                             'batch_size': analysis.best_config['batch_size'],
    #                             'include_hour': analysis.best_config['include_hour'],
    #                             }
    #                             , ignore_index=True)
# replace the layer_widths with the appropriate mapping using the layer_widths_dict
# layer_widths_dict = {'a':'[64,64]','b':'[128,128]','c':'[256,256]','d':'[64,128]','e':'[128,256]','f':'[256,512]','g':'[64,256]'}
# df_smape['layer_widths'] = df_smape['layer_widths'].apply(lambda x: layer_widths_dict[x])
# # save df_smape
# df_smape.to_csv('nbeats_smape.csv', index=False)