import pandas as pd
import numpy as np
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


class ScoutEngine:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.player_names = []

        # Clone engine
        self.scaler = None
        self.pca = None
        self.feature_cols = []
        self.knn_models = {}
        self.role_group_indices = {}
        self.pca_feature_map = None
        self.weighted_feature_matrix = None

        # Value engine
        self.value_model = None
        self.value_model_name = None
        self.best_xgb_params = {
            'n_estimators': 250,
            'max_depth': 3,
            'learning_rate': 0.07,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 4,
            'reg_alpha': 0.2,
            'reg_lambda': 1.0
        }
        self.value_feature_cols = []
        self.value_metrics = {}
        self.model_comparison = {}

        self.presets = self._get_presets()
        self.labels = self._get_labels()

    def load_data(self):
        try:
            try:
                raw_df = pd.read_csv(self.filepath, encoding='utf-8')
            except:
                raw_df = pd.read_csv(self.filepath, encoding='latin-1')

            min_minutes = 600

            if 'Min' in raw_df.columns:
                self.df = raw_df[raw_df['Min'] >= min_minutes].copy().fillna(0)
            else:
                self.df = raw_df.copy().fillna(0)

            self.df.columns = (
                self.df.columns
                .str.replace('_per90', '_p90', regex=False)
                .str.replace('per90', '_p90', regex=False)
            )

            # Do NOT convert percentages / ratios to per-90
            ignore = [
                'Player', 'Pos', 'Squad', 'Comp', 'Age', '90s', 'Min',
                'Born', 'Rk', 'market_value_in_eur', 'Season',
                'Won%', 'G/Sh', 'SoT%', 'Pass_Short_Cmp%', 'Pass_Med_Cmp%', 'Pass_Long_Cmp%'
            ]

            cols_to_convert = []
            for col in self.df.columns:
                if (
                    self.df[col].dtype in ['float64', 'int64']
                    and col not in ignore
                    and '_p90' not in col
                ):
                    if f'{col}_p90' not in self.df.columns:
                        cols_to_convert.append(col)

            if cols_to_convert and '90s' in self.df.columns:
                valid_mask = self.df['90s'] > 0
                p90_df = pd.DataFrame(index=self.df.index)

                for col in cols_to_convert:
                    new_col = f'{col}_p90'
                    p90_df[new_col] = 0.0
                    p90_df.loc[valid_mask, new_col] = (
                        self.df.loc[valid_mask, col] / self.df.loc[valid_mask, '90s']
                    )

                self.df = pd.concat([self.df, p90_df], axis=1)

            self.df.replace([np.inf, -np.inf], 0, inplace=True)

            if 'Player' in self.df.columns:
                self.player_names = sorted(self.df['Player'].dropna().unique().tolist())

            exclude = [
                'Player', 'Squad', 'Nation', 'Pos', 'Comp', 'Age', 'Born',
                '90s', 'Min', 'Rk', 'market_value_in_eur', 'Fair_Value',
                'Undervalued_Index', 'Season'
            ]

            self.feature_cols = [
                c for c in self.df.columns
                if self.df[c].dtype in ['float64', 'int64'] and c not in exclude
            ]

            self._train_clone_engine()
            self._train_value_model()
            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================
    def _normalize_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower().strip()
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return text

    # =========================================================
    # CLONE ENGINE
    # =========================================================
    def _get_role_group(self, pos_value):
        if pd.isna(pos_value):
            return 'OTHER'

        pos_value = str(pos_value)

        if 'DF' in pos_value:
            return 'DF'
        if 'MF' in pos_value:
            return 'MF'
        if 'FW' in pos_value:
            return 'FW'
        return 'OTHER'

    def _get_clone_role_weights(self):
        return {
            'DF': {
                'Tkl_p90': 10,
                'Int_p90': 10,
                'Blocks_p90': 9,
                'Clr_p90': 9,
                'Aerials_Won_p90': 8,
                'Won%': 7,
                'Recov_p90': 7,
                'PrgP_p90': 5,
                'Pass_Long_Cmp_p90': 5
            },
            'MF': {
                'PrgP_p90': 10,
                'PrgC_p90': 9,
                'KP_p90': 8,
                'SCA_p90': 8,
                'Pass_Short_Cmp_p90': 7,
                'Pass_Med_Cmp_p90': 7,
                'Pass_Long_Cmp_p90': 6,
                'Recov_p90': 6,
                'Int_p90': 5,
                'Tkl_p90': 5
            },
            'FW': {
                'npxG_p90': 10,
                'Gls_p90': 10,
                'Sh_p90': 8,
                'SoT_p90': 8,
                'SCA_p90': 7,
                'GCA_p90': 6,
                'PrgC_p90': 7,
                'Succ_p90': 7,
                'Touches_Att_Pen_p90': 8,
                'KP_p90': 4
            },
            'OTHER': {}
        }

    def _get_role_focus_features(self):
        return {
            'DF': [
                'Tkl_p90', 'Int_p90', 'Blocks_p90', 'Clr_p90',
                'Aerials_Won_p90', 'Won%', 'Recov_p90',
                'PrgP_p90', 'Pass_Long_Cmp_p90'
            ],
            'MF': [
                'PrgP_p90', 'PrgC_p90', 'KP_p90', 'SCA_p90',
                'Pass_Short_Cmp_p90', 'Pass_Med_Cmp_p90',
                'Pass_Long_Cmp_p90', 'Recov_p90', 'Int_p90', 'Tkl_p90'
            ],
            'FW': [
                'npxG_p90', 'Gls_p90', 'Sh_p90', 'SoT_p90',
                'SCA_p90', 'GCA_p90', 'PrgC_p90',
                'Succ_p90', 'Touches_Att_Pen_p90', 'KP_p90'
            ],
            'OTHER': self.feature_cols[:10]
        }

    def _build_weighted_clone_features(self):
        weighted_df = self.df[self.feature_cols].copy().astype(float)
        clone_weights = self._get_clone_role_weights()

        for col in weighted_df.columns:
            weighted_df[col] = weighted_df[col] * 1.0

        for idx, row in self.df.iterrows():
            role_group = row.get('Role_Group', 'OTHER')
            role_weights = clone_weights.get(role_group, {})

            for feature_name, weight in role_weights.items():
                if feature_name in weighted_df.columns:
                    weighted_df.at[idx, feature_name] = weighted_df.at[idx, feature_name] * weight

        return weighted_df

    def _train_clone_engine(self):
        self.df['Role_Group'] = self.df['Pos'].apply(self._get_role_group) if 'Pos' in self.df.columns else 'OTHER'

        self.weighted_feature_matrix = self._build_weighted_clone_features()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.weighted_feature_matrix)

        self.pca = PCA(n_components=0.95)
        X_pca = self.pca.fit_transform(X_scaled)

        self.pca_feature_map = pd.DataFrame(
            X_pca,
            index=self.df.index,
            columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
        )

        print(f"📉 PCA Reduced Features: {X_scaled.shape[1]} -> {X_pca.shape[1]}")

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

    def _similarity_from_cosine_distance(self, distances):
        similarities = (1 - distances) * 100
        similarities = np.clip(similarities, 0, 100)
        return similarities

    def _explain_clone_match(self, target_player, clone_player, top_k=3):
        role_group = target_player.get('Role_Group', 'OTHER')
        focus_features = self._get_role_focus_features().get(role_group, self.feature_cols[:10])

        valid_features = [f for f in focus_features if f in self.feature_cols]

        comparisons = []

        for feat in valid_features:
            target_val = float(target_player.get(feat, 0))
            clone_val = float(clone_player.get(feat, 0))

            diff = abs(target_val - clone_val)
            scale = max(abs(target_val), abs(clone_val), 1.0)
            normalized_diff = diff / scale

            comparisons.append({
                "feature": feat,
                "target_value": round(target_val, 3),
                "clone_value": round(clone_val, 3),
                "difference": round(diff, 3),
                "normalized_diff": normalized_diff
            })

        comparisons = sorted(comparisons, key=lambda x: x["normalized_diff"])

        top_matches = [
            {
                "feature": item["feature"],
                "target": item["target_value"],
                "clone": item["clone_value"]
            }
            for item in comparisons[:top_k]
        ]

        key_differences = [
            {
                "feature": item["feature"],
                "target": item["target_value"],
                "clone": item["clone_value"]
            }
            for item in comparisons[-top_k:]
        ]

        return top_matches, key_differences

    def find_clones(self, player_name, top_n=10, same_role_only=True):
        normalized_query = self._normalize_text(player_name)

        exact_mask = self.df['Player'].apply(lambda x: self._normalize_text(x) == normalized_query)
        matches = self.df[exact_mask]

        if matches.empty:
            contains_mask = self.df['Player'].apply(lambda x: normalized_query in self._normalize_text(x))
            matches = self.df[contains_mask]

        if matches.empty:
            suggestions = [
                p for p in self.player_names
                if normalized_query in self._normalize_text(p)
                or self._normalize_text(p).startswith(normalized_query[:3])
            ][:10]
            return {"error": "Player not found", "suggestions": suggestions}

        target_player = matches.iloc[0]
        target_idx = target_player.name
        target_role_group = target_player.get('Role_Group', 'OTHER')

        target_row = pd.DataFrame([target_player[self.feature_cols]], columns=self.feature_cols).astype(float)
        role_weights = self._get_clone_role_weights().get(target_role_group, {})

        for feature_name, weight in role_weights.items():
            if feature_name in target_row.columns:
                target_row[feature_name] = target_row[feature_name] * weight

        target_scaled = self.scaler.transform(target_row)
        target_pca = self.pca.transform(target_scaled)

        if same_role_only and target_role_group in self.knn_models:
            knn_model = self.knn_models[target_role_group]
            subset_idx = self.role_group_indices[target_role_group]
            distances, local_indices = knn_model.kneighbors(
                target_pca,
                n_neighbors=min(top_n + 5, len(subset_idx))
            )
            candidate_indices = subset_idx[local_indices[0]]
        else:
            global_knn = NearestNeighbors(
                n_neighbors=min(top_n + 5, len(self.df)),
                metric='cosine',
                algorithm='brute'
            )
            global_knn.fit(self.pca_feature_map.values)
            distances, local_indices = global_knn.kneighbors(
                target_pca,
                n_neighbors=min(top_n + 5, len(self.df))
            )
            candidate_indices = self.df.index[local_indices[0]]

        results = self.df.loc[candidate_indices].copy()
        results['Distance'] = distances[0]
        results['Similarity'] = self._similarity_from_cosine_distance(results['Distance'].values)

        results = results[results.index != target_idx]

        if 'Pos' in results.columns and 'Pos' in target_player.index:
            target_pos = str(target_player['Pos'])
            exact_pos_mask = results['Pos'].astype(str) == target_pos

            exact_pos_results = results[exact_pos_mask].copy()
            other_results = results[~exact_pos_mask].copy()

            results = pd.concat([exact_pos_results, other_results], axis=0)

        results = results.sort_values(by=['Similarity'], ascending=False).head(top_n)

        explained_results = []
        for idx, row in results.iterrows():
            top_matches, key_differences = self._explain_clone_match(target_player, row, top_k=3)

            row_dict = row.to_dict()
            row_dict["Top_Matches"] = top_matches
            row_dict["Key_Differences"] = key_differences

            explained_results.append(row_dict)

        return explained_results

    # =========================================================
    # VALUE ENGINE
    # =========================================================
    def _prepare_value_features(self, df):
        base = df[self.feature_cols + ['Age']].copy()

        if 'Comp' in df.columns:
            comp_dummies = pd.get_dummies(df['Comp'].fillna('Unknown'), prefix='Comp')
            base = pd.concat([base, comp_dummies], axis=1)

        return base

    def _evaluate_model(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        return mae, rmse, r2

    def _train_value_model(self):
        train_df = self.df[self.df['market_value_in_eur'] > 0].copy()

        if train_df.empty:
            print("❌ No valid rows with market_value_in_eur > 0")
            return

        if 'Player' not in train_df.columns:
            print("❌ 'Player' column is required for grouped train/test split.")
            return

        y = train_df['market_value_in_eur']
        groups = train_df['Player']

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(train_df, y, groups=groups))

        train_part = train_df.iloc[train_idx].copy()
        test_part = train_df.iloc[test_idx].copy()

        X_train = self._prepare_value_features(train_part)
        X_test = self._prepare_value_features(test_part)

        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        y_train = train_part['market_value_in_eur']
        y_test = test_part['market_value_in_eur']

        train_players = set(train_part['Player'])
        test_players = set(test_part['Player'])
        overlap = train_players.intersection(test_players)

        print("\n👥 Grouped Split by Player:")
        print(f"Train rows   : {len(X_train)}")
        print(f"Test rows    : {len(X_test)}")
        print(f"Train players: {len(train_players)}")
        print(f"Test players : {len(test_players)}")
        print(f"Overlap      : {len(overlap)} players")
        print(f"\n🧪 Value model feature count: {X_train.shape[1]}")

        self.value_feature_cols = X_train.columns.tolist()
        self.model_comparison = {}

        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        rf_mae, rf_rmse, rf_r2 = self._evaluate_model(rf_model, X_train, X_test, y_train, y_test)
        self.model_comparison["Random Forest"] = {
            "MAE": rf_mae,
            "RMSE": rf_rmse,
            "R2": rf_r2
        }

        print("\n📊 Model Evaluation:")
        print("\nRandom Forest:")
        print(f"MAE : €{rf_mae:,.2f}")
        print(f"RMSE: €{rf_rmse:,.2f}")
        print(f"R²  : {rf_r2:.4f}")

        xgb_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            **self.best_xgb_params
        )

        xgb_mae, xgb_rmse, xgb_r2 = self._evaluate_model(xgb_model, X_train, X_test, y_train, y_test)
        self.model_comparison["XGBoost_Final"] = {
            "MAE": xgb_mae,
            "RMSE": xgb_rmse,
            "R2": xgb_r2,
            "params": self.best_xgb_params
        }

        print("\nXGBoost_Final:")
        print(f"Params: {self.best_xgb_params}")
        print(f"MAE : €{xgb_mae:,.2f}")
        print(f"RMSE: €{xgb_rmse:,.2f}")
        print(f"R²  : {xgb_r2:.4f}")

        self.value_model_name = "XGBoost_Final"
        self.value_metrics = {
            "Best_Model": self.value_model_name,
            "Best_Model_MAE": xgb_mae,
            "Best_Model_RMSE": xgb_rmse,
            "Best_Model_R2": xgb_r2,
            "Train_Rows": len(X_train),
            "Test_Rows": len(X_test),
            "Train_Players": len(train_players),
            "Test_Players": len(test_players),
            "Player_Overlap": len(overlap),
            "Value_Feature_Count": X_train.shape[1],
            "Best_XGB_Params": self.best_xgb_params
        }

        print(f"\n🏆 Final Value Model: {self.value_model_name}")

        full_X = self._prepare_value_features(train_df)
        full_X = full_X.reindex(columns=self.value_feature_cols, fill_value=0)

        self.value_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            **self.best_xgb_params
        )

        self.value_model.fit(full_X, y)

        all_X = self._prepare_value_features(self.df)
        all_X = all_X.reindex(columns=self.value_feature_cols, fill_value=0)

        self.df['Fair_Value'] = self.value_model.predict(all_X)
        self.df['Fair_Value'] = self.df['Fair_Value'].clip(lower=0)
        self.df['Undervalued_Index'] = self.df.apply(self._calc_index, axis=1)

    # =========================================================
    # GENERAL METHODS
    # =========================================================
    def _calc_index(self, row):
        actual = row['market_value_in_eur']
        fair = row['Fair_Value']

        if actual <= 0 or fair <= actual:
            return 0

        ratio = ((fair - actual) / actual) * 100
        return min(ratio, 100)

    def get_player_list(self):
        return self.player_names

    def get_config(self):
        return {
            "presets": self.presets,
            "labels": self.labels,
            "features": self.feature_cols,
            "value_metrics": self.value_metrics,
            "model_comparison": self.model_comparison,
            "best_model": self.value_model_name,
            "best_xgb_params": self.best_xgb_params,
            "value_feature_count": len(self.value_feature_cols)
        }

    def attribute_search(self, weights, role, budget, max_age):
        target = self.df.copy()

        if 'Age' in target.columns:
            target = target[target['Age'] <= max_age]

        if 'Pos' in target.columns:
            if 'Back' in role:
                target = target[target['Pos'].str.contains('DF', na=False)]
            elif 'Mid' in role:
                target = target[target['Pos'].str.contains('MF', na=False)]
            elif 'Striker' in role or 'Winger' in role:
                target = target[target['Pos'].str.contains('FW', na=False)]

        if 'market_value_in_eur' in target.columns:
            limit = budget * 1_000_000
            target = target[
                (target['market_value_in_eur'] <= limit) |
                (target['market_value_in_eur'].isna())
            ]

        ranked = pd.DataFrame(index=target.index)
        valid_weights = {}

        for k, v in weights.items():
            col = k if k in target.columns else f"{k}_p90"
            if col in target.columns:
                ranked[col] = target[col].rank(pct=True)
                valid_weights[col] = v

        if not valid_weights:
            return []

        scores = np.zeros(len(target))
        total_w = sum(valid_weights.values())

        for k, w in valid_weights.items():
            scores += ranked[k] * w

        target['Scout_Score'] = (scores / total_w) * 100
        result_df = target.sort_values(by='Scout_Score', ascending=False).head(50)

        return result_df.to_dict(orient='records')

    def _get_presets(self):
        return {
            'Center Back (Ball Playing)': {
                'PrgP_p90': 9, 'Pass_Into_1_3_p90': 8, 'Aerials_Won_p90': 7, 'Int_p90': 7, 'Tkl_p90': 5
            },
            'Center Back (Stopper)': {
                'Aerials_Won_p90': 10, 'Clr_p90': 9, 'Blocks_p90': 8, 'TklW_p90': 7, 'Won%': 6
            },
            'Full Back (Attacking)': {
                'PrgC_p90': 9, 'Crs_p90': 8, 'SCA90': 7, 'Tkl_p90': 6, 'Int_p90': 5
            },
            'Defensive Mid (Destroyer)': {
                'Tkl_p90': 10, 'Int_p90': 9, 'Blocks_p90': 8, 'Recov_p90': 7, 'Pass_Short_Cmp_p90': 5
            },
            'Deep Lying Playmaker': {
                'PrgP_p90': 10, 'Pass_Into_1_3_p90': 9, 'Pass_Long_Cmp_p90': 8, 'Int_p90': 6, 'KP_p90': 5
            },
            'Box-to-Box Midfielder': {
                'PrgC_p90': 8, 'PrgP_p90': 8, 'Tkl_p90': 7, 'SCA90': 7, 'Recov_p90': 7
            },
            'Attacking Mid (Creator)': {
                'SCA90': 10, 'KP_p90': 9, 'Pass_Into_Box_p90': 8, 'Succ_p90': 7, 'PrgC_p90': 7
            },
            'Winger (Dribbler)': {
                'Succ_p90': 10, 'PrgC_p90': 9, 'Touches_Att_Pen_p90': 8, 'SCA90': 7, 'npxG_p90': 6
            },
            'Striker (Complete)': {
                'npxG_p90': 9, 'Sh_p90': 8, 'SCA90': 8, 'PrgP_p90': 7, 'Aerials_Won_p90': 6
            },
            'Striker (Poacher)': {
                'npxG_p90': 10, 'SoT_p90': 9, 'Touches_Att_Pen_p90': 9, 'Gls_p90': 8, 'G/Sh': 7
            }
        }

    def _get_labels(self):
        return {
            'npxG_p90': 'Non-Pen xG',
            'Gls_p90': 'Goals',
            'Ast_p90': 'Assists',
            'SCA90': 'Shot Creating Actions',
            'PrgP_p90': 'Progressive Passes',
            'PrgC_p90': 'Progressive Carries',
            'Tkl_p90': 'Tackles',
            'Int_p90': 'Interceptions',
            'market_value_in_eur': 'Market Value (€)',
            'Undervalued_Index': 'Undervalued Score (0-100)'
        }
