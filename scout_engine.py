import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA  # <--- NEW IMPORT

class ScoutEngine:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.player_names = []
        self.ml_model = None       # KNN
        self.value_model = None    # Random Forest
        self.scaler = None
        self.pca = None            # <--- PCA Model
        self.feature_cols = []
        self.presets = self._get_presets()
        self.labels = self._get_labels()

    def load_data(self):
        try:
            # Load and clean data
            raw_df = pd.read_csv(self.filepath)
            min_minutes = 600
            
            if 'Min' in raw_df.columns:
                self.df = raw_df[raw_df['Min'] >= min_minutes].copy().fillna(0)
            else:
                self.df = raw_df.copy().fillna(0)

            # Standardize column names
            self.df.columns = self.df.columns.str.replace('_per90', '_p90').str.replace('per90', '_p90')

            # Calculate p90 stats if missing
            cols_to_convert = []
            ignore = ['Player', 'Pos', 'Squad', 'Comp', 'Age', '90s', 'Min', 'Born', 'Rk', 'market_value_in_eur']
            for col in self.df.columns:
                if self.df[col].dtype in ['float64', 'int64'] and col not in ignore and '_p90' not in col:
                    if f'{col}_p90' not in self.df.columns:
                        cols_to_convert.append(col)

            if cols_to_convert and '90s' in self.df.columns:
                p90_df = self.df[cols_to_convert].div(self.df['90s'], axis=0).add_suffix('_p90')
                self.df = pd.concat([self.df, p90_df], axis=1)

            self.df.replace([np.inf, -np.inf], 0, inplace=True)
            self.player_names = sorted(self.df['Player'].unique().tolist())

            # Prepare ML Features
            exclude = ['Player', 'Squad', 'Nation', 'Pos', 'Comp', 'Age', 'Born', '90s', 'Min', 'Rk', 'market_value_in_eur', 'Fair_Value', 'Undervalued_Index']
            self.feature_cols = [c for c in self.df.columns if self.df[c].dtype in ['float64', 'int64'] and c not in exclude]

            # Train Models
            self._train_models()
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def _train_models(self):
        # --- 1. Clone Engine (PCA + KNN) ---
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.df[self.feature_cols])
        
        # Apply PCA to keep 95% of variance
        self.pca = PCA(n_components=0.95)
        X_pca = self.pca.fit_transform(X_scaled)
        
        print(f"📉 PCA Reduced Features: {X_scaled.shape[1]} -> {X_pca.shape[1]}")
        
        # Train KNN on compressed data
        self.ml_model = NearestNeighbors(n_neighbors=15, algorithm='ball_tree')
        self.ml_model.fit(X_pca)

        # --- 2. Value Engine (Random Forest) ---
        train_df = self.df[self.df['market_value_in_eur'] > 0].copy()
        if not train_df.empty:
            X_val = train_df[self.feature_cols + ['Age']]
            y_val = train_df['market_value_in_eur']
            
            # Random Forest handles raw data well, so we use X_val (not PCA)
            self.value_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.value_model.fit(X_val, y_val)
            
            # Predict for all
            all_X = self.df[self.feature_cols + ['Age']]
            self.df['Fair_Value'] = self.value_model.predict(all_X)
            
            # Calculate Index
            self.df['Undervalued_Index'] = self.df.apply(self._calc_index, axis=1)

    def _calc_index(self, row):
        actual = row['market_value_in_eur']
        fair = row['Fair_Value']
        if actual <= 0 or fair <= actual: return 0
        ratio = ((fair - actual) / actual) * 100
        return min(ratio, 100)

    def get_player_list(self):
        return self.player_names

    def get_config(self):
        return {
            "presets": self.presets,
            "labels": self.labels,
            "features": self.feature_cols
        }

    def attribute_search(self, weights, role, budget, max_age):
        target = self.df.copy()
        
        # Filters
        if 'Age' in target.columns: target = target[target['Age'] <= max_age]
        if 'Pos' in target.columns:
            if 'Back' in role: target = target[target['Pos'].str.contains('DF', na=False)]
            elif 'Mid' in role: target = target[target['Pos'].str.contains('MF', na=False)]
            elif 'Striker' in role or 'Winger' in role: target = target[target['Pos'].str.contains('FW', na=False)]
            
        if 'market_value_in_eur' in target.columns:
            limit = budget * 1_000_000
            target = target[(target['market_value_in_eur'] <= limit) | (target['market_value_in_eur'].isna())]

        # Scoring
        ranked = pd.DataFrame(index=target.index)
        valid_weights = {}
        for k, v in weights.items():
            col = k if k in target.columns else f"{k}_p90"
            if col in target.columns:
                ranked[col] = target[col].rank(pct=True)
                valid_weights[col] = v
        
        if not valid_weights: return []

        scores = np.zeros(len(target))
        total_w = sum(valid_weights.values())
        for k, w in valid_weights.items(): scores += ranked[k] * w
            
        target['Scout_Score'] = (scores / total_w) * 100
        result_df = target.sort_values(by='Scout_Score', ascending=False).head(50)
        
        # Format for API
        return result_df.to_dict(orient='records')

    def find_clones(self, player_name):
        matches = self.df[self.df['Player'] == player_name]
        
        # Fallback search
        if matches.empty:
            matches = self.df[self.df['Player'].str.contains(player_name, case=False, na=False)]
            
        if matches.empty: return {"error": "Player not found"}
        
        target_player = matches.iloc[0]
        target_data = target_player[self.feature_cols].values.reshape(1, -1)
        
        # Pipeline: Scale -> PCA -> Search
        target_scaled = self.scaler.transform(target_data)
        target_pca = self.pca.transform(target_scaled) # <--- Transform target to PCA space
        
        distances, indices = self.ml_model.kneighbors(target_pca)
        
        results = self.df.iloc[indices[0]].copy()
        
        # Calculate Similarity %
        max_dist = distances.max() if distances.max() > 0 else 1
        results['Similarity'] = (1 - (distances[0] / (max_dist * 1.5))) * 100
        results['Similarity'] = results['Similarity'].clip(0, 100)
        
        # Format for API
        return results.to_dict(orient='records')

    def _get_presets(self):
        return {
            'Center Back (Ball Playing)': {'PrgP_p90': 9, 'Pass_Into_1_3_p90': 8, 'Aerials_Won_p90': 7, 'Int_p90': 7, 'Tkl_p90': 5},
            'Center Back (Stopper)': {'Aerials_Won_p90': 10, 'Clr_p90': 9, 'Blocks_p90': 8, 'TklW_p90': 7, 'Won%': 6},
            'Full Back (Attacking)': {'PrgC_p90': 9, 'Crs_p90': 8, 'SCA90': 7, 'Tkl_p90': 6, 'Int_p90': 5},
            'Defensive Mid (Destroyer)': {'Tkl_p90': 10, 'Int_p90': 9, 'Blocks_p90': 8, 'Recov_p90': 7, 'Pass_Short_Cmp_p90': 5},
            'Deep Lying Playmaker': {'PrgP_p90': 10, 'Pass_Into_1_3_p90': 9, 'Pass_Long_Cmp_p90': 8, 'Int_p90': 6, 'KP_p90': 5},
            'Box-to-Box Midfielder': {'PrgC_p90': 8, 'PrgP_p90': 8, 'Tkl_p90': 7, 'SCA90': 7, 'Recov_p90': 7},
            'Attacking Mid (Creator)': {'SCA90': 10, 'KP_p90': 9, 'Pass_Into_Box_p90': 8, 'Succ_p90': 7, 'PrgC_p90': 7},
            'Winger (Dribbler)': {'Succ_p90': 10, 'PrgC_p90': 9, 'Touches_Att_Pen_p90': 8, 'SCA90': 7, 'npxG_p90': 6},
            'Striker (Complete)': {'npxG_p90': 9, 'Sh_p90': 8, 'SCA90': 8, 'PrgP_p90': 7, 'Aerials_Won_p90': 6},
            'Striker (Poacher)': {'npxG_p90': 10, 'SoT_p90': 9, 'Touches_Att_Pen_p90': 9, 'Gls_p90': 8, 'G/Sh': 7}
        }

    def _get_labels(self):
        return {
            'npxG_p90': 'Non-Pen xG', 'Gls_p90': 'Goals', 'Ast_p90': 'Assists',
            'SCA90': 'Shot Creating Actions', 'PrgP_p90': 'Progressive Passes',
            'PrgC_p90': 'Progressive Carries', 'Tkl_p90': 'Tackles', 
            'Int_p90': 'Interceptions', 'market_value_in_eur': 'Market Value (€)',
            'Undervalued_Index': 'Undervalued Score (0-100)'
        }
