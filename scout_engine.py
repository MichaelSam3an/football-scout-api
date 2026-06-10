import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


class ScoutEngine:
    def __init__(
        self,
        filepath,
        reduction_method="pca",
        value_reduction="pca",
        clone_components=20
    ):
        self.filepath = filepath
        self.clone_components = clone_components

        self.reduction_method = reduction_method.lower()
        self.value_reduction = value_reduction.lower()

        self.df = None
        self.player_names = []

        self.cluster_knn_models = {}
        self.cluster_indices = {}

        # Clone engine
        self.scaler = None
        self.clone_reducer = None
        self.feature_cols = []
        self.knn_models = {}
        self.role_group_indices = {}
        self.clone_reducer_feature_map = None
        self.weighted_feature_matrix = None

        # Value engine
        self.value_model = None
        self.value_model_name = None

        self.best_xgb_params = {
            'n_estimators': 400,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'reg_alpha': 0.1,
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

            ELITE_CLUBS = [
                "Real Madrid",
                "Barcelona",
                "Manchester City",
                "Liverpool",
                "Arsenal",
                "Chelsea",
                "Manchester United",
                "Bayern Munich",
                "Paris S-G",
                "Atletico Madrid",
            ]

            STRONG_CLUBS = [
                "Inter",
                "Milan",
                "Juventus",
                "Dortmund",
                "Leverkusen",
                "Newcastle",
                "Napoli",
                "Roma",
                "Lazio",
                "Marseille",
                "Benfica",
                "Porto"
            ]

            def get_club_tier(squad):
                if squad in ELITE_CLUBS:
                    return 3
                elif squad in STRONG_CLUBS:
                    return 2
                return 1

            self.df["Club_Tier"] = self.df["Squad"].apply(get_club_tier)
            self.df["Age_squared"] = self.df["Age"] ** 2

            self.df["Elite_Club"] = (
                self.df["Squad"]
                .isin(ELITE_CLUBS)
                .astype(int)
            )


            if 'Player' in self.df.columns:
                self.player_names = sorted(self.df['Player'].dropna().unique().tolist())

            exclude = [
                'Player', 'Squad', 'Nation', 'Pos', 'Comp', 'Age', 'Born',
                '90s', 'Min', 'Rk', 'market_value_in_eur', 'Fair_Value',
                'Undervalued_Index', 'Season'
            ]
            duplicates = self.df.columns[self.df.columns.duplicated()]
            print("Duplicate columns:", duplicates.tolist())

            self.feature_cols = []

            for c in self.df.columns:

                try:

                    if (
                        self.df[c].dtype in ['float64', 'int64']
                        and c not in exclude
                    ):
                        self.feature_cols.append(c)

                except Exception as e:

                    print("Problem column:", c)
                    print(type(self.df[c]))
                    print(e)

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
        X_scaled = self.scaler.fit_transform(
            self.weighted_feature_matrix
        )

        self.clone_reducer = TruncatedSVD(
            n_components=min(
                self.clone_components,
                X_scaled.shape[1] - 1
            ),
            algorithm="randomized",
            random_state=42
        )

        X_reduced = self.clone_reducer.fit_transform(X_scaled)

        retained_var = np.sum(
            self.clone_reducer.explained_variance_ratio_
        )

        print(
            f"\n⚙ Clone Reduction: "
            f"{self.reduction_method.upper()}"
        )

        print(
            f"Features Reduced: "
            f"{X_scaled.shape[1]} -> {X_reduced.shape[1]}"
        )

        print(
            f"Variance Retained: "
            f"{retained_var:.4f}"
        )

        print(
            f"📊 {self.reduction_method.upper()} Variance Retained: "
            f"{retained_var:.4f}"
        )

        self.clone_reducer_feature_map = pd.DataFrame(
            X_reduced,
            index=self.df.index,
            columns=[f'PC{i+1}' for i in range(X_reduced.shape[1])]
        )

        self.cluster_knn_models = {}
        self.cluster_indices = {}
        self.df["Role_Cluster"] = -1

        for role_group in self.df["Role_Group"].dropna().unique():
            subset_idx = self.df[self.df["Role_Group"] == role_group].index

            if len(subset_idx) < 8:
                continue

            subset_vectors = self.clone_reducer_feature_map.loc[subset_idx].values

            n_clusters = min(4, max(2, len(subset_idx) // 20))
            n_clusters = min(n_clusters, len(subset_idx))

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=20
            )

            labels = kmeans.fit_predict(subset_vectors)
            self.df.loc[subset_idx, "Role_Cluster"] = labels

            for label in np.unique(labels):
                cluster_idx = subset_idx[labels == label]

                if len(cluster_idx) >= 3:
                    cluster_vectors = self.clone_reducer_feature_map.loc[cluster_idx].values

                    knn = NearestNeighbors(
                        n_neighbors=min(20, len(cluster_idx)),
                        metric='cosine',
                        algorithm='brute'
                    )
                    knn.fit(cluster_vectors)

                    self.cluster_knn_models[(role_group, int(label))] = knn
                    self.cluster_indices[(role_group, int(label))] = cluster_idx

        self.knn_models = {}
        self.role_group_indices = {}

        for role_group in self.df['Role_Group'].dropna().unique():
            subset_idx = self.df[self.df['Role_Group'] == role_group].index

            if len(subset_idx) >= 3:
                subset_vectors = self.clone_reducer_feature_map.loc[subset_idx].values

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
        target_pca = self.clone_reducer.transform(target_scaled)

        target_cluster = int(target_player.get("Role_Cluster", -1))
        cluster_key = (target_role_group, target_cluster)

        if same_role_only and cluster_key in self.cluster_knn_models:
            knn_model = self.cluster_knn_models[cluster_key]
            subset_idx = self.cluster_indices[cluster_key]
            distances, local_indices = knn_model.kneighbors(
                target_pca,
                n_neighbors=min(top_n + 5, len(subset_idx))
            )
            candidate_indices = subset_idx[local_indices[0]]

        elif same_role_only and target_role_group in self.knn_models:
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
            global_knn.fit(self.clone_reducer_feature_map.values)
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
        base = df[
            self.feature_cols +
            [
                'Age',
                'Age_squared',
                'Club_Tier',
                'Elite_Club'
            ]
        ].copy()
        base['Age_Cubed'] = base['Age'] ** 2

        if 'npxG_p90' in df.columns:
            base['Age_x_npxG'] = (
                base['Age'] * df['npxG_p90']
            )

        if 'Gls_p90' in df.columns:
            base['Age_x_Gls'] = (
                base['Age'] * df['Gls_p90']
            )
        base['Is_Peak_Age'] = (
            (base['Age'] >= 23)
            & (base['Age'] <= 28)
        ).astype(int)


       # NEW
        big_nations = [
            'ENG',
            'BRA',
            'ARG',
            'FRA',
            'ESP',
            'POR',
            'GER',
            'ITA',
            'NED'
        ]

        if 'Nation' in df.columns:

            base['Big_Football_Nation'] = (
                df['Nation']
                .astype(str)
                .str.upper()
                .str.contains(
                    '|'.join(big_nations),
                    regex=True
                )
            ).astype(int)


        if 'Comp' in df.columns:
            comp_dummies = pd.get_dummies(
                df['Comp'].fillna('Unknown'),
                prefix='Comp'
            )

            base = pd.concat(
                [base, comp_dummies],
                axis=1
            )

        # NEW
        if 'Squad' in df.columns:

            squad_dummies = pd.get_dummies(
                df['Squad'].fillna('Unknown'),
                prefix='Squad'
            )

            base = pd.concat(
                [base, squad_dummies],
                axis=1
            )

        if 'Pos' in df.columns:

            pos_dummies = pd.get_dummies(
                df['Pos'].fillna('Unknown'),
                prefix='Pos'
            )

            base = pd.concat(
                [base, pos_dummies],
                axis=1
            )


        base = base.loc[:, ~base.columns.duplicated()]
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

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
        train_idx, test_idx = next(gss.split(train_df, y, groups=groups))

        train_part = train_df.iloc[train_idx].copy()
        test_part = train_df.iloc[test_idx].copy()

        X_train = self._prepare_value_features(train_part)
        X_test = self._prepare_value_features(test_part)

        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
        self.original_value_columns = X_train.columns.tolist()
        # =====================================
        # VALUE MODEL DIMENSIONALITY REDUCTION
        # =====================================

        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        self.value_scaler = StandardScaler()

        X_train_scaled = self.value_scaler.fit_transform(X_train)
        X_test_scaled = self.value_scaler.transform(X_test)

        if self.value_reduction == "pca":

            self.value_pca = PCA(
                n_components=0.985,
                random_state=42
            )

        elif self.value_reduction == "rsvd":

            self.value_pca = TruncatedSVD(
                n_components=min(
                    120,
                    X_train_scaled.shape[1] - 1
                ),
                algorithm="randomized",
                random_state=42
            )

        elif self.value_reduction == "none":

            self.value_pca = None

        else:

            raise ValueError(
                "value_reduction must be "
                "'none', 'pca', or 'rsvd'"
            )

        if self.value_pca is not None:

            X_train = self.value_pca.fit_transform(
                X_train_scaled
            )

            X_test = self.value_pca.transform(
                X_test_scaled
            )

            print(
                f"\n⚙ Value Model Reduction: "
                f"{self.value_reduction.upper()}"
            )

            print(
                f"Features Reduced: "
                f"{X_train_scaled.shape[1]} -> {X_train.shape[1]}"
            )

        else:

            X_train = X_train_scaled
            X_test = X_test_scaled

            print("\n⚙ Value Model Reduction: NONE")
        print(
            f"Features Reduced: "
            f"{X_train_scaled.shape[1]} -> {X_train.shape[1]}"
        )


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

        if isinstance(X_train, pd.DataFrame):
            self.value_feature_cols = X_train.columns.tolist()
        else:
            self.value_feature_cols = [
                f"PC{i+1}"
                for i in range(X_train.shape[1])
            ]

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
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        # Save for analysis
        self.y_test = y_test.copy()
        self.y_pred = pd.Series(y_pred, index=y_test.index)

        xgb_mae = mean_absolute_error(y_test, y_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        xgb_r2 = r2_score(y_test, y_pred)

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


        duplicates = full_X.columns[full_X.columns.duplicated()]
        print("FULL_X DUPLICATES:", duplicates.tolist())
        full_X = full_X.reindex(
            columns=self.original_value_columns,
            fill_value=0
        )

        full_X_scaled = self.value_scaler.transform(full_X)

        if self.value_pca is not None:
            full_X = self.value_pca.transform(full_X_scaled)
        else:
            full_X = full_X_scaled

        self.value_model = XGBRegressor(
                    objective='reg:squarederror',
                    random_state=42,
                    n_jobs=-1,
                    **self.best_xgb_params
                )

        self.value_model.fit(full_X, y)

        all_X = self._prepare_value_features(self.df)


        duplicates = all_X.columns[all_X.columns.duplicated()]
        print("ALL_X DUPLICATES:", duplicates.tolist())
        all_X = all_X.reindex(
            columns=self.original_value_columns,
            fill_value=0
        )

        all_X_scaled = self.value_scaler.transform(all_X)

        if self.value_pca is not None:
            all_X = self.value_pca.transform(all_X_scaled)
        else:
            all_X = all_X_scaled

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


    def get_dashboard_data(self):
        df = self.df.copy()

        # KPIs
        total_players = len(df)
        total_goals = int(df['Gls'].sum())
        eff_df = df[df['npxG'] >= 1.0].copy()
        eff_df['_ratio'] = eff_df['Gls'] / eff_df['npxG']
        clinical_count = int((eff_df['_ratio'] > 1.2).sum())
        avg_xg_efficiency = round(float(eff_df['_ratio'].mean()), 2) if len(eff_df) > 0 else 0.0

        cols_base = ['Player', 'Squad', 'Pos']

        top_scorers = df.nlargest(5, 'Gls')[cols_base + ['Gls', 'npxG']].round(2).to_dict('records')
        top_assisters = df.nlargest(5, 'Ast')[cols_base + ['Ast']].round(2).to_dict('records')

        # Most Clinical (min npxG 2.0)
        clin_df = df[df['npxG'] >= 2.0].copy()
        clin_df['finisher_ratio'] = (clin_df['Gls'] / clin_df['npxG']).round(2)
        clin_df['finisher_badge'] = clin_df['finisher_ratio'].apply(
            lambda r: 'Clinical' if r > 1.2 else ('Average' if r >= 0.8 else 'Wasteful')
        )
        most_clinical = clin_df.nlargest(5, 'finisher_ratio')[
            cols_base + ['Gls', 'npxG', 'finisher_ratio', 'finisher_badge', 'market_value_in_eur']
        ].round(2).to_dict('records')

        # Hidden Gems (min market_value > 0)
        gems_df = df[df['market_value_in_eur'] > 0]
        hidden_gems = gems_df.nlargest(5, 'Undervalued_Index')[
            cols_base + ['Age', 'market_value_in_eur', 'Fair_Value', 'Undervalued_Index']
        ].round(2).to_dict('records')

        # Team Goals vs xG (top 15 by goals)
        team_agg = df.groupby('Squad').agg(goals=('Gls', 'sum'), xg=('npxG', 'sum')).reset_index()
        team_agg['goals'] = team_agg['goals'].astype(int)
        team_agg['xg'] = team_agg['xg'].round(1)
        team_goals_vs_xg = team_agg.nlargest(15, 'goals').to_dict('records')

        return {
            'kpis': {
                'total_players': total_players,
                'total_goals': total_goals,
                'clinical_count': clinical_count,
                'avg_xg_efficiency': avg_xg_efficiency,
            },
            'top_scorers': top_scorers,
            'top_assisters': top_assisters,
            'most_clinical': most_clinical,
            'hidden_gems': hidden_gems,
            'team_goals_vs_xg': team_goals_vs_xg,
        }

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
