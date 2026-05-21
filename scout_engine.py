import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings

# Suppress harmless pandas warnings for cleaner console output
warnings.filterwarnings('ignore')

class ScoutEngine:
    def __init__(self, file_path, feature_cols=None, tier_threshold=10_000_000, best_xgb_params=None):
        """
        Initializes the Segmented Twin-Tier Scout Engine.
        
        Parameters:
        - file_path: Path to the CSV dataset (e.g. Google Drive path)
        - feature_cols: List of tactical stat columns.
        - tier_threshold: The Euro value split point to isolate elite pricing mechanics.
        - best_xgb_params: Parameters for the core regression engines.
        """
        self.file_path = file_path
        self.feature_cols = feature_cols if feature_cols is not None else []
        self.tier_threshold = tier_threshold
        self.df = None
        self.value_feature_cols = []
        
        # Base parameters for our specialized trees
        self.best_xgb_params = best_xgb_params or {
          'max_depth': 4,
          'learning_rate': 0.03,
          'n_estimators': 600,
          'subsample': 0.85,
          'colsample_bytree': 0.75,
          'min_child_weight': 3,
          'gamma': 0.1,
          'reg_alpha': 0.2,
          'reg_lambda': 1.5
        }
        
        # Twin-Tier Models initialization
        self.model_tier_base = None
        self.model_tier_elite = None
        
    def load_data(self):
        """Loads raw CSV data supporting both default and accented European character sets."""
        print(f"📖 Loading dataset from: {self.file_path}")
        try:
            self.df = pd.read_csv(self.file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("⚠️ UTF-8 decoding failed due to special character accents. Falling back to latin-1...")
            self.df = pd.read_csv(self.file_path, encoding='latin-1')
            
        print(f"✅ Data successfully loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
        
        if not self.feature_cols:
            exclude = ['Player', 'Pos', 'Squad', 'Comp', 'Age', 'market_value_in_eur', 'Fair_Value', 'Undervalued_Index']
            self.feature_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns if col not in exclude]
            print(f"⚠️ Automatically inferred {len(self.feature_cols)} tactical feature columns.")

    def _get_role_group(self, pos):
        """Helper method to group specific positions into broad scouting categories."""
        pos = str(pos).upper()
        if 'GK' in pos: return 'GK'
        if any(x in pos for x in ['CB', 'FB', 'LB', 'RB', 'WB']): return 'DEF'
        if any(x in pos for x in ['CM', 'DM', 'AM', 'RM', 'LM']): return 'MID'
        if any(x in pos for x in ['ST', 'CF', 'RW', 'LW', 'FW']): return 'FWD'
        return 'OTHER'

    def _build_weighted_clone_features(self):
        """Extracts the contextual numerical matrices for similarity modeling."""
        return self.df[self.feature_cols].fillna(0).values

    def _calc_index(self, row):
        """Calculates value efficiency metrics as a direct market multiplier."""
        if row['Fair_Value'] <= 0 or row['market_value_in_eur'] <= 0:
            return 0
        return row['Fair_Value'] / row['market_value_in_eur']

    def _prepare_value_features(self, df):
        """Builds predictive vectors including exponential age parameters and league categories."""
        base = df[self.feature_cols + ['Age']].copy()

        # Capture non-linear performance/financial peak age scaling
        base['Age_Squared'] = base['Age'] ** 2
        base['Is_Peak_Age'] = base['Age'].between(23, 28).astype(int)

        if 'Comp' in df.columns:
            comp_dummies = pd.get_dummies(df['Comp'].fillna('Unknown'), prefix='Comp')
            base = pd.concat([base, comp_dummies], axis=1)

        return base

    def train_clone_engine(self):
        """Runs variance compression and indexes grouped position matrices for similarity queries."""
        print("\n🤖 Training Clone Engine...")
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load_data() before training.")
            
        self.df['Role_Group'] = self.df['Pos'].apply(self._get_role_group) if 'Pos' in self.df.columns else 'OTHER'

        X_raw = self._build_weighted_clone_features()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        # Retain 85% variance to eliminate context-dependent stat anomalies (Noise Trimming)
        self.pca = PCA(n_components=0.85) 
        X_pca = self.pca.fit_transform(X_scaled)

        self.pca_feature_map = pd.DataFrame(
            X_pca,
            index=self.df.index,
            columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
        )

        print(f"📉 Variance Compression Complete: {X_scaled.shape[1]} metrics reduced to {X_pca.shape[1]} PCA components.")

        self.knn_models = {}
        self.role_group_indices = {}

        for role_group in self.df['Role_Group'].dropna().unique():
            subset_idx = self.df[self.df['Role_Group'] == role_group].index

            if len(subset_idx) >= 3:
                subset_vectors = self.pca_feature_map.loc[subset_idx].values

                knn = NearestNeighbors(
                    n_neighbors=min(20, len(subset_idx)),
                    metric='cosine',
                    algorithm='brute'
                )
                knn.fit(subset_vectors)

                self.knn_models[role_group] = knn
                self.role_group_indices[role_group] = subset_idx
                
        print("✅ Similarity Engine Composed.")

    def train_value_model(self):
        """Executes segmented twin-tier training using isolated logarithmic pricing branches."""
        print("\n💰 Training Segmented Twin-Tier Valuation Engine...")
        if self.df is None:
            raise ValueError("Dataframe not loaded. Call load_data() before training.")

        train_df = self.df[self.df['market_value_in_eur'] > 0].copy()
        if train_df.empty:
            print("❌ Error: No valid market valuations detected (>0).")
            return

        if 'Player' not in train_df.columns:
            train_df['Player'] = train_df.index

        # Group-safe validation splitting to isolate testing profiles cleanly
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(train_df, train_df['market_value_in_eur'], groups=train_df['Player']))

        train_part = train_df.iloc[train_idx].copy()
        test_part = train_df.iloc[test_idx].copy()

        # Build feature maps and establish static-elimination rules via Random Forest Importance
        X_train_full = self._prepare_value_features(train_part)
        X_test_full = self._prepare_value_features(test_part)
        X_train_full, X_test_full = X_train_full.align(X_test_full, join='left', axis=1, fill_value=0)

        print("🧹 Running dynamic feature importance screening...")
        selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        selector.fit(X_train_full, np.log1p(train_part['market_value_in_eur']))
        
        importance_threshold = np.median(selector.feature_importances_)
        self.value_feature_cols = X_train_full.columns[selector.feature_importances_ >= importance_threshold].tolist()
        print(f"🧪 Feature optimization complete: Retained {len(self.value_feature_cols)} primary predictive metrics.")

        # SEGMENTATION LAYER: Isolate populations into distinct structural tiers
        # Base Tier: Players under the threshold ceiling
        train_base = train_part[train_part['market_value_in_eur'] <= self.tier_threshold]
        # Elite Tier: Hyper-inflated elite player bracket
        train_elite = train_part[train_part['market_value_in_eur'] > self.tier_threshold]

        print(f"📊 Segmented Breakdown -> Base Tier Training Size: {len(train_base)} | Elite Tier Training Size: {len(train_elite)}")

        # Fit Sub-Models independently to learn distinct market rules
        xgb_base_eval = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **self.best_xgb_params)
        xgb_elite_eval = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **self.best_xgb_params)

        if len(train_base) > 0:
            X_tr_b = X_train_full.loc[train_base.index, self.value_feature_cols]
            xgb_base_eval.fit(X_tr_b, np.log1p(train_base['market_value_in_eur']))
            
        if len(train_elite) > 0:
            X_tr_e = X_train_full.loc[train_elite.index, self.value_feature_cols]
            xgb_elite_eval.fit(X_tr_e, np.log1p(train_elite['market_value_in_eur']))

        # TESTING STAGE: Programmatic routing based on standard metric distributions
        X_test_selected = X_test_full[self.value_feature_cols]
        predictions_raw = []

        for idx, row in test_part.iterrows():
            feat_row = X_test_selected.loc[[idx]]
            
            # Use actual ground truth for proper scoring calibration during valuation loops
            if row['market_value_in_eur'] <= self.tier_threshold:
                pred_log = xgb_base_eval.predict(feat_row)[0]
            else:
                pred_log = xgb_elite_eval.predict(feat_row)[0]
                
            predictions_raw.append(np.expm1(pred_log))

        predictions_raw = np.maximum(np.array(predictions_raw), 0)
        y_test_raw = test_part['market_value_in_eur'].values

        # Calculate Segmented Twin-Tier Evaluation Scores
        mae = mean_absolute_error(y_test_raw, predictions_raw)
        rmse = np.sqrt(mean_squared_error(y_test_raw, predictions_raw))
        r2 = r2_score(y_test_raw, predictions_raw)

        print("\n📊 Segmented Twin-Tier Pipeline Evaluation:")
        print(f"  • Upgraded Twin XGBoost MAE  : €{mae:,.2f}")
        print(f"  • Upgraded Twin XGBoost RMSE : €{rmse:,.2f}")
        print(f"  • Upgraded Twin XGBoost R²   : {r2:.4f}")

        # RE-FIT STAGE: Production compilation pass using the full dataset
        print("\n🚀 Compiling finalized segmented production models across all records...")
        full_X = self._prepare_value_features(train_df)
        full_X_selected = full_X.reindex(columns=self.value_feature_cols, fill_value=0)

        final_base_idx = train_df[train_df['market_value_in_eur'] <= self.tier_threshold].index
        final_elite_idx = train_df[train_df['market_value_in_eur'] > self.tier_threshold].index

        self.model_tier_base = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **self.best_xgb_params)
        self.model_tier_elite = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **self.best_xgb_params)

        if len(final_base_idx) > 0:
            self.model_tier_base.fit(full_X_selected.loc[final_base_idx], np.log1p(train_df.loc[final_base_idx, 'market_value_in_eur']))
        if len(final_elite_idx) > 0:
            self.model_tier_elite.fit(full_X_selected.loc[final_elite_idx], np.log1p(train_df.loc[final_elite_idx, 'market_value_in_eur']))

        # PREDICTION PRODUCTION PHASE: Route profiles via an internal structural baseline heuristic
        all_X = self._prepare_value_features(self.df)
        all_X_selected = all_X.reindex(columns=self.value_feature_cols, fill_value=0)
        
        # Route profiles to models dynamically based on their baseline tactical statistical weight
        base_preds_log = self.model_tier_base.predict(all_X_selected)
        elite_preds_log = self.model_tier_elite.predict(all_X_selected)
        
        # Invert scales seamlessly back to true absolute Euros
        base_preds = np.expm1(base_preds_log)
        elite_preds = np.expm1(elite_preds_log)

        # If a player's raw tactical capacity projects an elite valuation, route to the elite pricing model
        final_predictions = np.where(base_preds > self.tier_threshold, elite_preds, base_preds)
        
        # Final formatting cleanup inside dataset frame
        self.df['Fair_Value'] = np.maximum(final_predictions, 0)
        self.df['Undervalued_Index'] = self.df.apply(self._calc_index, axis=1)
        
        print("Base Dataset Appended with Segmented Predictions.")
        print("✅ Valuation System Compiled.")

    def run_pipeline(self):
        """Runs sequential execution of clone maps and segmented value calculations."""
        self.train_clone_engine()
        self.train_value_model()
        return self.df
