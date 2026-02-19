import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sksurv.nonparametric import kaplan_meier_estimator
from tqdm import tqdm, trange
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines import CoxTimeVaryingFitter

# own imports
from src.utils import create_nfl_field
from src.utils import load_data
from src.utils import life_expectancy

np.set_printoptions(precision=4)
pd.set_option("display.precision", 4)

if __name__ == "__main__":
    print(os.getcwd())
    games, plays, players, scouting, plays_with_collapse, off_cols, def_cols = (
        load_data()
    )
    print(plays_with_collapse.shape)
    print(plays_with_collapse.columns)
    players_with_collapse = pd.merge(
        scouting[scouting.pff_role.isin(["Pass", "Pass Rush", "Pass Block"])][
            ["gameId", "playId", "nflId", "pff_role"]
        ],
        plays_with_collapse,
        how="right",
        on=["gameId", "playId"],
    )

    off_team_cols = [
        "possessionTeam_ATL",
        "possessionTeam_BAL",
        "possessionTeam_BUF",
        "possessionTeam_CAR",
        "possessionTeam_CHI",
        "possessionTeam_CIN",
        "possessionTeam_CLE",
        "possessionTeam_DAL",
        "possessionTeam_DEN",
        "possessionTeam_DET",
        "possessionTeam_GB",
        "possessionTeam_HOU",
        "possessionTeam_IND",
        "possessionTeam_JAX",
        "possessionTeam_KC",
        "possessionTeam_LA",
        "possessionTeam_LAC",
        "possessionTeam_LV",
        "possessionTeam_MIA",
        "possessionTeam_MIN",
        "possessionTeam_NE",
        "possessionTeam_NO",
        "possessionTeam_NYG",
        "possessionTeam_NYJ",
        "possessionTeam_PHI",
        "possessionTeam_PIT",
        "possessionTeam_SEA",
        "possessionTeam_SF",
        "possessionTeam_TB",
        "possessionTeam_TEN",
        "possessionTeam_WAS",
    ]

    def_team_cols = [
        "defensiveTeam_ATL",
        "defensiveTeam_BAL",
        "defensiveTeam_BUF",
        "defensiveTeam_CAR",
        "defensiveTeam_CHI",
        "defensiveTeam_CIN",
        "defensiveTeam_CLE",
        "defensiveTeam_DAL",
        "defensiveTeam_DEN",
        "defensiveTeam_DET",
        "defensiveTeam_GB",
        "defensiveTeam_HOU",
        "defensiveTeam_IND",
        "defensiveTeam_JAX",
        "defensiveTeam_KC",
        "defensiveTeam_LA",
        "defensiveTeam_LAC",
        "defensiveTeam_LV",
        "defensiveTeam_MIA",
        "defensiveTeam_MIN",
        "defensiveTeam_NE",
        "defensiveTeam_NO",
        "defensiveTeam_NYG",
        "defensiveTeam_NYJ",
        "defensiveTeam_PHI",
        "defensiveTeam_PIT",
        "defensiveTeam_SEA",
        "defensiveTeam_SF",
        "defensiveTeam_TB",
        "defensiveTeam_TEN",
        "defensiveTeam_WAS",
    ]

    ## subset of feature columns
    base_feat = [
        "yardsToGo",
        "defendersInBox",
        "Cover-0",
        "Cover-1",
        "Cover-2",
        "Cover-6",
        "misc_def",
        "Quarters",  # cover 4
        "2-Man",  # "cover 5"ish
        "pff_playAction",
        "down2",
        "down3",
        "shotgun",
    ]
    opt_feat = ["mean_polydists"] + [f"poly_tau_{i}" for i in range(5, 41, 5)]

    # columns that need to be normalized (maybe?)
    norm_cols = list(
        set(base_feat + opt_feat).intersection(
            ["mean_polydists", "defendersInBox", "yardsToGo"]
            + [f"poly_tau_{i}" for i in range(5, 41, 5)]
        )
    )

    print("build training and val set...")
    plays_with_collapse["misc_def"] = plays_with_collapse.pff_passCoverage.isin(
        ["Red Zone", "Bracket", "Prevent", "Miscellaneous"]
    )

    X = pd.get_dummies(
        plays_with_collapse[base_feat + opt_feat + ["defensiveTeam", "possessionTeam"]],
        columns=["defensiveTeam", "possessionTeam"],
    )
    X[["defensiveTeam", "possessionTeam"]] = plays_with_collapse[
        ["defensiveTeam", "possessionTeam"]
    ]

    X[norm_cols] = (X[norm_cols] - X[norm_cols].mean()) / X[norm_cols].std()
    Y = plays_with_collapse[["survived", "surv_frame", "pass_frame"]]

    X["time"] = Y[["surv_frame", "pass_frame"]].min(axis=1)
    X["collapse_event"] = (~Y["survived"]) & (Y.surv_frame.lt(Y.pass_frame))
    X["pass_event"] = Y.surv_frame.gt(Y.pass_frame)

    X_train, X_test, Y_train, Y_test = train_test_split(X.index, Y.index, test_size=0.2)
    Xs = X.iloc[X_train]
    Xts = X.iloc[X_test]

    data_df = pd.concat([Xs, Y.iloc[Y_train]], axis=1)
    data_df["collapse_event"] = (~data_df["survived"]) & (
        data_df.surv_frame.lt(data_df.pass_frame)
    )
    data_df["pass_event"] = data_df.surv_frame.gt(data_df.pass_frame)
    data_df["time"] = data_df[["surv_frame", "pass_frame"]].min(axis=1)

    test_data_df = pd.concat([Xts, Y.iloc[Y_test]], axis=1)
    test_data_df["collapse_event"] = (~test_data_df["survived"]) & (
        test_data_df.surv_frame.lt(test_data_df.pass_frame)
    )
    test_data_df["pass_event"] = test_data_df.surv_frame.gt(test_data_df.pass_frame)
    test_data_df["time"] = test_data_df[["surv_frame", "pass_frame"]].min(axis=1)

    stratas = ["shotgun", "pff_playAction"]
    pass_concordances = []
    collapse_concordances = []
    verbose = 0
    print("Condition baseline COX PH")
    cond = [[], *[[coli] for coli in opt_feat]] + [
        [
            "poly_tau_10",
            "poly_tau_20",
            "poly_tau_30",
        ],
        [
            "poly_tau_20",
            "poly_tau_25",
            "poly_tau_30",
        ],
    ]
    for add_feat in tqdm(cond):
        if verbose > 0:
            print(add_feat)
            print("pass timing")
            print(data_df[base_feat + add_feat + ["pass_event", "time"]].columns)
            print(
                data_df[base_feat + add_feat + ["pass_event", "time"]].corr(
                    numeric_only=True
                )
            )
        pass_cox = CoxPHFitter(strata=stratas)
        pass_cox.fit(
            data_df[base_feat + add_feat + ["pass_event", "time"]],
            duration_col="time",
            event_col="pass_event",
            robust=True,
        )
        if verbose > 1:
            pass_cox.print_summary()
            pass_cox.check_assumptions(
                data_df[base_feat + add_feat + ["pass_event", "time"]]
            )

        pass_concordances.append(pass_cox.score(test_data_df, "concordance_index"))

        if verbose > 0:
            print("collapse timing")
        collapse_cox = CoxPHFitter(strata=stratas)
        collapse_cox.fit(
            data_df[base_feat + add_feat + ["collapse_event", "time"]],
            duration_col="time",
            event_col="collapse_event",
            robust=True,
        )
        if verbose > 1:
            collapse_cox.print_summary()
        collapse_cox.check_assumptions(
            data_df[base_feat + add_feat + ["collapse_event", "time"]]
        )
        collapse_concordances.append(
            collapse_cox.score(test_data_df, "concordance_index")
        )

    print(
        pd.DataFrame(
            {
                "feat": cond,
                "pass": pass_concordances,
                "collapse": collapse_concordances,
            }
        )
    )
