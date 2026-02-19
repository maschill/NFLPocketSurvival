import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sksurv.nonparametric import kaplan_meier_estimator
from tqdm import tqdm, trange
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines import CoxTimeVaryingFitter

# own imports
from src.utils import create_nfl_field
from src.utils import load_data
from src.utils import life_expectancy


if __name__ == "__main__":
    print(os.getcwd())
    games, plays, players, scouting, plays_with_collapse, off_cols, def_cols = (
        load_data()
    )
    print(plays_with_collapse.shape)

    # region KM plot
    plt.figure(figsize=(12, 6))
    x, y, conf_int = kaplan_meier_estimator(
        (
            plays_with_collapse[plays_with_collapse.surv_frame.notna()].poly_frame > -1
        ).to_numpy(),
        plays_with_collapse[
            plays_with_collapse.surv_frame.notna()
        ].surv_frame.to_numpy(),
        conf_type="log-log",
    )
    le = life_expectancy(x, y)
    plt.step(x, y, where="post", label=f"all {le:3.2f}")
    plt.fill_between(x, conf_int[0], conf_int[1], alpha=0.25, step="post")

    x, y, conf_int = kaplan_meier_estimator(
        (
            plays_with_collapse[
                plays_with_collapse.surv_frame.notna()
                & (plays_with_collapse.defensiveTeam == "CLE")
            ].poly_frame
            > -1
        ).to_numpy(),
        plays_with_collapse[
            plays_with_collapse.surv_frame.notna()
            & (plays_with_collapse.defensiveTeam == "CLE")
        ].surv_frame.to_numpy(),
        conf_type="log-log",
    )
    le = life_expectancy(x, y)
    plt.step(x, y, where="post", label=f"Browns {le:3.2f}")
    plt.fill_between(x, conf_int[0], conf_int[1], alpha=0.25, step="post")

    plt.xlabel("pocket survival frames")
    plt.ylabel("survival rate")
    plt.title("Frames Until the Pocket Collapses Against Clevland Browns Pass Rush")
    plt.legend()
    plt.savefig("example_km_plot.png", dpi=600)

    print(
        "compute the naive life expectancy of the pass block (pocket collapse)(>50 snaps)"
    )

    players_with_collapse = pd.merge(
        scouting[scouting.pff_role.isin(["Pass", "Pass Rush", "Pass Block"])][
            ["gameId", "playId", "nflId", "pff_role"]
        ],
        plays_with_collapse,
        how="right",
        on=["gameId", "playId"],
    )

    ple_off = []
    for plyr in tqdm(
        players_with_collapse[
            players_with_collapse.pff_role == "Pass Block"
        ].nflId.unique()
    ):
        tdf = players_with_collapse[
            players_with_collapse.surv_frame.notna()
            & (players_with_collapse.nflId == plyr)
        ]
        x, y, conf_int = kaplan_meier_estimator(
            (tdf.poly_frame > -1).to_numpy(),
            tdf.surv_frame.to_numpy(),
            conf_type="log-log",
        )
        le = life_expectancy(x, y)
        ple_off.append(
            {
                "nflId": plyr,
                "life expectancy": le,
                "num_snaps": tdf.shape[0],
                "team": tdf.possessionTeam.iat[0],
            }
        )
    pb_df = (
        pd.DataFrame(ple_off)
        .merge(players, on="nflId")
        .drop(columns=["height", "weight", "collegeName", "birthDate"])
    )

    print("Top 5 Centers")
    print(
        pb_df[(pb_df.num_snaps > 50) & (pb_df.officialPosition == "C")]
        .sort_values("life expectancy", ascending=False)
        .head(5)
    )
    print("Top 5 offensive Tackles")
    print(
        pb_df[(pb_df.num_snaps > 50) & (pb_df.officialPosition == "T")]
        .sort_values("life expectancy", ascending=False)
        .head(5)
    )
    print("Top 5 Guards")
    print(
        pb_df[(pb_df.num_snaps > 50) & (pb_df.officialPosition == "G")]
        .sort_values("life expectancy", ascending=False)
        .head(5)
    )
    print("#" * 80)
    print("and the naive life expectancy of the pass rush (>50 snaps)")
    ple_def = []
    for plyr in tqdm(
        players_with_collapse[
            players_with_collapse.pff_role == "Pass Rush"
        ].nflId.unique()
    ):
        tdf = players_with_collapse[
            players_with_collapse.surv_frame.notna()
            & (players_with_collapse.nflId == plyr)
            & (players_with_collapse.pff_role == "Pass Rush")
        ]
        x, y, conf_int = kaplan_meier_estimator(
            (tdf.poly_frame > -1).to_numpy(),
            tdf.surv_frame.to_numpy(),
            conf_type="log-log",
        )
        le = life_expectancy(x, y)
        ple_def.append(
            {
                "nflId": plyr,
                "life expectancy": le,
                "num_snaps": tdf.shape[0],
                "team": tdf.defensiveTeam.iat[0],
            }
        )
    pr_df = (
        pd.DataFrame(ple_def)
        .merge(players, on="nflId")
        .drop(columns=["height", "weight", "collegeName", "birthDate"])
    )

    print("Top 5 Defensive Ends")
    print(
        pr_df[(pr_df.num_snaps > 50) & (pr_df.officialPosition == "DE")]
        .sort_values("life expectancy")
        .head(5)
    )
    print("Top 5 Defensive Tackles")
    print(
        pr_df[(pr_df.num_snaps > 50) & (pr_df.officialPosition == "DT")]
        .sort_values("life expectancy")
        .head(5)
    )
    print("Top 5 Nose Tackles")
    print(
        pr_df[(pr_df.num_snaps > 50) & (pr_df.officialPosition == "NT")]
        .sort_values("life expectancy")
        .head(5)
    )
    print("#" * 80)
    print("Top 5 Linebackeres")
    print(
        pr_df[
            (pr_df.num_snaps > 50)
            & (pr_df.officialPosition.isin(["OLB", "LB", "MLB", "ILB"]))
        ]
        .sort_values("life expectancy")
        .head(5)
    )
    print("#" * 80)
    print("Top 5 DBs")
    print(
        pr_df[
            (pr_df.num_snaps > 50) & (pr_df.officialPosition.isin(["CB", "SS", "FS"]))
        ]
        .sort_values("life expectancy")
        .head(5)
    )
    print("#" * 80)
