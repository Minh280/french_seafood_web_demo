# import os
# import joblib
# import pandas as pd
# from hierarchicalforecast.core import HierarchicalReconciliation
# from hierarchicalforecast.methods import MinTrace
# import numpy as np

# class SeafoodForecaster:
#     def __init__(self, model_dir="model_artifacts"):
#         print(f"INFO: Loading artifacts from {model_dir}...")
        
#         # Load Model
#         fcst_path = os.path.join(model_dir, "fcst_model.pkl")
#         self.fcst = joblib.load(fcst_path)
        
#         self.fcst.n_jobs = 1 

#         self.S_df = joblib.load(os.path.join(model_dir, "S_df.pkl"))
#         self.tags = joblib.load(os.path.join(model_dir, "tags.pkl"))
        
#         self.Y_fitted_ensemble = joblib.load(os.path.join(model_dir, "Y_fitted_ensemble.pkl"))
#         print(self.Y_fitted_ensemble.shape)

#         self.hrec = HierarchicalReconciliation(
#             reconcilers=[MinTrace(method="mint_shrink")]
#         )
#         self.weights = {"HW": 0.6, "ARIMA": 0.4}


#         print("INFO: Warming up model (JIT compilation)...")
#         _ = self.fcst.predict(h=1)
#         print("INFO: Ready.")

#     def predict(self, horizon: int):
#         Y_hat_df = self.fcst.predict(h=horizon)
#         Y_hat_df['Ensemble'] = (
#             Y_hat_df['HW'] * self.weights['HW'] + 
#             Y_hat_df['ARIMA'] * self.weights['ARIMA']
#         )
#         print("Predicted Successfully")

        
#         if 'unique_id' not in Y_hat_df.columns:
#             Y_hat_df = Y_hat_df.reset_index()

#         available_cols = Y_hat_df.columns.tolist()
#         hw = "HW" if "HW" in available_cols else "HoltWinters"
#         arima = "ARIMA" if "ARIMA" in available_cols else "AutoARIMA"
        

#         if hw not in available_cols or arima not in available_cols:
#              pass

        
#         Y_hat_ens = Y_hat_df[['unique_id', 'ds', 'Ensemble']]
#         Y_fitted_ens = self.Y_fitted_ensemble[['unique_id', 'ds', 'y', 'Ensemble']]

#         # Reconciliation
#         Y_rec_df = self.hrec.reconcile(
#             Y_hat_df=Y_hat_ens,
#             Y_df=Y_fitted_ens,
#             S_df=self.S_df,
#             tags=self.tags
#         )

#         num_cols = Y_rec_df.select_dtypes(include=[np.number]).columns
#         Y_rec_df[num_cols] = Y_rec_df[num_cols].clip(lower=0)

#         target_col = "Ensemble/MinTrace_method-mint_shrink"

#         return (
#             Y_rec_df[["unique_id", "ds", target_col]]
#             .rename(columns={target_col: "y_pred"})
#         )

# if __name__ == "__main__":
#     model_dir = "artifacts"
#     forecaster = SeafoodForecaster(model_dir=model_dir)
#     print(forecaster.predict(horizon=12))


import os
import joblib
import pandas as pd
import numpy as np
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace

class SeafoodForecaster:
    def __init__(self, model_dir="model_artifacts"):
        print(f"INFO: Loading artifacts from {model_dir}...")
        
        # 1. Load Model
        fcst_path = os.path.join(model_dir, "fcst_model.pkl")
        self.fcst = joblib.load(fcst_path)
        self.fcst.n_jobs = 1 

        self.S_df = joblib.load(os.path.join(model_dir, "S_df.pkl"))
        self.tags = joblib.load(os.path.join(model_dir, "tags.pkl"))

        self.Y_fitted_ensemble = joblib.load(os.path.join(model_dir, "Y_fitted_ensemble.pkl"))
        self.Y_fitted_ensemble['ds'] = pd.to_datetime(self.Y_fitted_ensemble['ds'])
        
        full_path = os.path.join(model_dir, "Y_full.pkl")
        self.Y_full = joblib.load(full_path)
        self.Y_full['ds'] = pd.to_datetime(self.Y_full['ds'])

        self.last_train_date = self.Y_fitted_ensemble['ds'].max()
        self.last_actual_date = self.Y_full['ds'].max()
        
        test_dates = self.Y_full[self.Y_full['ds'] > self.last_train_date]['ds'].unique()
        self.test_horizon = len(test_dates)
        
        print(f"INFO: Train ends: {self.last_train_date.date()}")
        print(f"INFO: Full data ends: {self.last_actual_date.date()}")
        print(f"INFO: Gap to bridge (Test set): {self.test_horizon} steps")

        # 5. Setup Reconciliation
        self.hrec = HierarchicalReconciliation(
            reconcilers=[MinTrace(method="mint_shrink")]
        )
        self.weights = {"HW": 0.6, "ARIMA": 0.4}

        # Warm-up
        try:
            _ = self.fcst.predict(h=1)
        except:
            pass

    def predict(self, horizon: int):
        """
        Dự báo tổng lực: Bù đắp khoảng Test (Gap) + Tương lai (Horizon)
        """
        total_horizon = self.test_horizon + horizon
        
        Y_hat_df = self.fcst.predict(h=total_horizon)
        if 'unique_id' not in Y_hat_df.columns:
            Y_hat_df = Y_hat_df.reset_index()

        cols = Y_hat_df.columns
        hw = "HW" if "HW" in cols else "HoltWinters"
        arima = "ARIMA" if "ARIMA" in cols else "AutoARIMA"

        Y_hat_df['Ensemble'] = (
            Y_hat_df[hw] * self.weights['HW'] + 
            Y_hat_df[arima] * self.weights['ARIMA']
        )

        Y_hat_ens = Y_hat_df[['unique_id', 'ds', 'Ensemble']]
        
        Y_fitted_ens = self.Y_fitted_ensemble[['unique_id', 'ds', 'y', 'Ensemble']]

        Y_rec_df = self.hrec.reconcile(
            Y_hat_df=Y_hat_ens,
            Y_df=Y_fitted_ens,
            S_df=self.S_df,
            tags=self.tags
        )

        num_cols = Y_rec_df.select_dtypes(include=[np.number]).columns
        Y_rec_df[num_cols] = Y_rec_df[num_cols].clip(lower=0)

        target_col = "Ensemble/MinTrace_method-mint_shrink"
        if target_col not in Y_rec_df.columns:
             target_col = [c for c in Y_rec_df.columns if "MinTrace" in c][0]

        return Y_rec_df[["unique_id", "ds", target_col]].rename(columns={target_col: "y_pred"})

    def get_visualization_data(self, horizon: int):
        # A. Chạy dự báo
        df_pred_full = self.predict(horizon)
        df_pred_full['ds'] = pd.to_datetime(df_pred_full['ds'])

        # B. Phân loại TRAIN và TEST ACTUAL (Dựa vào Y_full)
        df_history = self.Y_full[['unique_id', 'ds', 'y']].copy()
        df_history['type'] = df_history['ds'].apply(
            lambda x: 'TRAIN' if x <= self.last_train_date else 'TRAIN'
        )
        
        df_pred_full['type'] = df_pred_full['ds'].apply(
            lambda x: 'TEST_PRED' if x <= self.last_actual_date else 'FUTURE'
        )
        df_pred_full = df_pred_full.rename(columns={'y_pred': 'y'})

        return {
            "history": df_history.to_dict(orient='records'),
            "prediction": df_pred_full.to_dict(orient='records')
        }

if __name__=="__main__":
    model_dir = "artifacts/model"
    forecaster = SeafoodForecaster(model_dir=model_dir)
    
    data = forecaster.get_visualization_data(horizon=4)
    
    print("Example Data (First 2 rows of prediction):")
    print(data['prediction'])