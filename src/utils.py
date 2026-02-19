import numpy as np
import pandas as pd

from scipy.spatial import Voronoi
from scipy.spatial import distance
import shapely
from shapely.geometry import Point, Polygon

from tqdm import tqdm, trange

tqdm.pandas()


# region HELPERS
def create_nfl_field() -> np.ndarray:
    """creates a simplified image of a football field as a background for rendering

    Returns:
        np.ndarray: the image in form of a np.array of the shape (x, y, 3)
    """
    empty_field = np.zeros((533, 1200, 3), dtype=np.int16) + np.array([50, 130, 20])
    empty_field[:, 0:100] = 127
    empty_field[:, 1100:] = 127
    empty_field[:, 90:101] = 255
    empty_field[:, 1100:1111] = 255
    for i in range(150, 1051, 50):
        empty_field[:, i - 2 : i + 3] = 255
    for i in range(110, 1091, 10):
        empty_field[:10, i - 1 : i + 2] = 255
        empty_field[-10:, i - 1 : i + 2] = 255
        empty_field[197:208, i - 1 : i + 2] = 255
        empty_field[327:338, i - 1 : i + 2] = 255
    return empty_field


def snap_filter(df):
    """
    use to add new column that is True whenever ball was snapped and NaN or False otherwise
    """
    if "ball_snap" in df.event.unique():
        frame = df.loc[df.event == "ball_snap", "frameId"].iat[0]
        df["is_snapped"] = df.frameId.ge(frame)

        return df.groupby("frameId").agg({"is_snapped": "max"})
    else:
        df["is_snapped"] = 1
        return df.groupby("frameId").agg({"is_snapped": "max"})


def add_snapped_filter_col(df):
    """add boolean indicating when ball was snapped"""
    snap_df = (
        df[["gameId", "playId", "nflId", "frameId", "event"]]
        .groupby(["gameId", "playId"])
        .apply(snap_filter)
        .reset_index()
    )
    if "is_snapped" in df.columns:
        df = df.drop(
            columns="is_snapped"
        )  # easiest way codewise, else merge will rename cols
    df = pd.merge(
        df,
        snap_df[["gameId", "playId", "frameId", "is_snapped"]],
        on=["gameId", "playId", "frameId"],
    )
    return df


def calc_qb_space(fdf: pd.DataFrame, qb_clip0=0, qb_clip1=55) -> dict:
    """Compute the safe space around the QB. Is supposed to be used during df.groupby operations

    Args:
        fdf (pd.DataFrame): pandas dataframe containing tracking data and scouting data.
        qb_clip0(float) = Lower bound for qb y-coordinate. Defaults to 0.
        qb_clip1(float) = Upper bound for qb y-coordinate. Defaults to 55.

    Returns:
        Dictionary containing distance to closest rusher polygon, distance to clostest
        polygon corner, and binary flag whether qb is considered in danger.

    """
    points = np.stack(
        [
            fdf[fdf.pff_role.isin(["Pass Block", "Pass Rush"])].x.to_numpy(),
            fdf[fdf.pff_role.isin(["Pass Block", "Pass Rush"])].y.to_numpy(),
        ],
        axis=-1,
    )
    points = np.append(
        points, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0
    )
    vor = Voronoi(points)

    def_line_regions = (
        fdf[fdf.pff_role.isin(["Pass Block", "Pass Rush"])].pff_role == "Pass Rush"
    ).to_numpy()
    regions = vor.point_region[np.concatenate([def_line_regions, [False] * 4])]

    if len(regions) < 1:
        return {"qb_radius": 5.0, "qb_danger": False, "polydist": 5}

    def_pts = [
        [vor.vertices[vert] for vert in vor.regions[reg] if vert > -1]
        for reg in regions
        if len(vor.regions[reg]) > 0
    ]
    def_pts = np.concatenate(
        [np.stack(dpts, axis=0) for dpts in def_pts if len(dpts) > 0]
    )
    def_polys = []
    for reg in regions:
        if len(vor.regions[reg]) > 2:
            def_polys.append(Polygon([vor.vertices[vert] for vert in vor.regions[reg]]))

    qb_coords = np.stack(
        [
            fdf[fdf.pff_role == "Pass"].x.to_numpy(),
            np.clip(fdf[fdf.pff_role == "Pass"].y.to_numpy(), qb_clip0, qb_clip1),
        ]
    ).T
    radius = distance.cdist(qb_coords, def_pts).min()
    qb_danger = def_line_regions[distance.cdist(qb_coords, points).argmin()]

    if len(def_polys) < 1:
        print(fdf.gameId.iat[0], fdf.playId.iat[0], fdf.frameId.iat[0])
        return {"qb_radius": radius, "qb_danger": qb_danger, "polydist": 5}

    qb_pt = Point(qb_coords)
    polydist = min([poly.distance(qb_pt) for poly in def_polys])

    return {"qb_radius": radius, "qb_danger": qb_danger, "polydist": polydist}


def mean_polydist(df):
    return df.polydist.mean()


def mean_polydiff(df):
    return df.polydist.diff().mean()


def pass_frame(df):
    if df.event.isin(["pass_forward", "autoevent_passforward"]).sum() > 0:
        return (
            df[
                (df.event == "pass_forward") | (df.event == "autoevent_passforward")
            ].frameId.min()
            - df[df.is_snapped == 1].frameId.min()
        )
    return -(df.frameId.max() - df[df.is_snapped == 1].frameId.min())


def min_frame(df):
    return df.frameId.min()


def poly_frame(df):
    if df.polydist.min() > 0.001:
        return -1
    return df[df.polydist <= 0.001].frameId.min() - df.frameId.min()


def surv_frame(df):
    if df.polydist.min() > 0.0:
        return df.frameId.max() - df.frameId.min()
    return df[df.polydist == 0.0].frameId.min() - df.frameId.min()


def poly_tau(df, delta: int = 5):
    snapframe = df.frameId.min()
    offset = min(snapframe + delta, df.frameId.max())
    return df[df.frameId == offset].polydist.iat[0]


def pblockwin(df):
    snap = df.frameId.min()
    if df.polydist.min() < (0.001):
        collapse = df[df.polydist.le(0.001)].frameId.min()
        return collapse - snap >= 25
    return True


# https://stackoverflow.com/questions/45033980/how-to-compute-aic-for-linear-regression-model-in-python
def llf_(y, X, pr):
    # return maximized log likelihood
    nobs = float(X.shape[0])
    nobs2 = nobs / 2.0
    nobs = float(nobs)
    resid = y - pr
    ssr = np.sum((resid) ** 2)
    llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2
    return llf


def calc_aic(y, X, pr, p):
    # return aic metric
    llf = llf_(y, X, pr)
    return -2 * llf + 2 * p


# life expectancy
def life_expectancy(x, y):
    auc = x[0] * y[0]
    for i in range(1, len(x)):
        auc += (x[i] - x[i - 1]) * y[i]
    return auc


# region DATA LOAD
def load_data(base_path="nfl-big-data-bowl-2023/"):
    """Read the necessary data files and return the prepared dataframes

    Args:
        base_path (str, optional): Path to directory containing the data. Defaults to "nfl-big-data-bowl-2023/".

    Returns:
        _type_: data stuff
    """
    scouting = pd.read_parquet(f"{base_path}/scouting.parquet", engine="fastparquet")
    games = pd.read_parquet(f"{base_path}/games.parquet", engine="fastparquet")
    players = pd.read_parquet(f"{base_path}/players.parquet", engine="fastparquet")
    plays = pd.read_parquet(f"{base_path}/plays.parquet", engine="fastparquet")

    try:
        week = pd.read_parquet(f"{base_path}/tracking.parquet", engine="fastparquet")
        week_wscout = pd.read_parquet(
            f"{base_path}/tracking_wscout.parquet", engine="fastparquet"
        )
    except FileNotFoundError:  # if parquet file does not exists.
        weeks = [pd.read_csv(f"{base_path}/week{i}.csv") for i in trange(1, 9)]
        week = pd.concat(
            [add_snapped_filter_col(wk) for wk in weeks], ignore_index=True
        )
        week.to_parquet(f"{base_path}/tracking.parquet", engine="fastparquet")

        week_wscout = pd.merge(
            week[week.nflId.notna()], scouting, on=["gameId", "playId", "nflId"]
        )
        week.to_parquet(f"{base_path}/tracking_wscout.parquet", engine="fastparquet")
    nflverse_pbp = pd.read_parquet(
        f"{base_path}/play_by_play_2021.parquet",
        engine="fastparquet",
        columns=[
            "play_id",
            "old_game_id",
            "touchdown",
            "air_yards",
            "ep",
            "epa",
            "qb_epa",
            "xyac_epa",
            "xyac_mean_yardage",
            "xyac_success",
            "xyac_fd",
            "xpass",
            "pass_oe",
        ],
    )
    nflverse_pbp["old_game_id"] = pd.to_numeric(
        nflverse_pbp["old_game_id"], downcast="integer"
    )
    nflverse_pbp["play_id"] = pd.to_numeric(nflverse_pbp["play_id"], downcast="integer")
    plays = pd.merge(
        plays,
        nflverse_pbp,
        how="left",
        left_on=["gameId", "playId"],
        right_on=["old_game_id", "play_id"],
    )

    try:
        qb_spaces = pd.read_parquet(f"{base_path}/qb_spaces_pre_noclip.parquet")
    except FileNotFoundError:
        qb_spaces = (
            week_wscout[week_wscout.is_snapped == 1]
            .groupby(["gameId", "playId", "frameId"])
            .progress_apply(calc_qb_space)
            .reset_index()
        )
        qb_spaces["qb_radius"] = qb_spaces[0].progress_apply(lambda x: x["qb_radius"])
        qb_spaces["qb_danger"] = qb_spaces[0].progress_apply(lambda x: x["qb_danger"])
        qb_spaces["polydist"] = qb_spaces[0].progress_apply(lambda x: x["polydist"])
        qb_spaces = qb_spaces.drop(columns=0)
        qb_spaces.to_parquet(f"{base_path}/qb_spaces_pre_noclip.parquet")

    plays_with_collapse, off_cols, def_cols = prepare_plays_df(
        qb_spaces, week_wscout, plays
    )
    return games, plays, players, scouting, plays_with_collapse, off_cols, def_cols


# region PLAYS WITH COLLAPSE PREP
def prepare_plays_df(qb_spaces, week_wscout, plays):
    """computes the survivalframe and pass frame for each play

    Args:
        qb_spaces (_type_): df containing the pocket geometry
        week_wscout (_type_): df containing the tracking data
        plays (_type_): df containing the play meta data

    Returns:
        _type_: df with survival info, list of offensive df cols, list of defensive cols
    """
    pass_abs_frames = (
        week_wscout[
            (week_wscout.event == "pass_forward")
            | ((week_wscout.event == "autoevent_passforward"))
        ]
        .drop_duplicates(["gameId", "playId"])[["gameId", "playId", "frameId"]]
        .rename(columns={"frameId": "pass_absolute_frame"})
    )

    poly_times = (
        qb_spaces.groupby(["gameId", "playId"])
        .apply(poly_frame)
        .reset_index()
        .rename(columns={0: "poly_frame"})
    )

    min_frames = (
        qb_spaces.groupby(["gameId", "playId"])
        .apply(min_frame)
        .reset_index()
        .rename(columns={0: "min_frame"})
    )

    pass_frames = (
        week_wscout.groupby(["gameId", "playId"])
        .apply(pass_frame)
        .reset_index()
        .rename(columns={0: "pass_frame"})
    )

    surv_time = (
        qb_spaces.groupby(["gameId", "playId"])
        .progress_apply(surv_frame)
        .reset_index()
        .rename(columns={0: "surv_frame"})
    )

    pwin = (
        qb_spaces.groupby(["gameId", "playId"])
        .progress_apply(pblockwin)
        .reset_index()
        .rename(columns={0: "p_blockwin"})
    )

    mean_polydists = (
        qb_spaces.groupby(["gameId", "playId"])
        .polydist.mean()
        .reset_index()
        .rename(columns={"polydist": "mean_polydists"})
    )

    mean_polydiffs = (
        qb_spaces.groupby(["gameId", "playId"])
        .progress_apply(mean_polydiff)
        .reset_index()
        .rename(columns={0: "polydiff"})
    )

    timed_pocketsizes = [
        qb_spaces.groupby(["gameId", "playId"])
        .progress_apply(lambda x: poly_tau(x, i))
        .reset_index()
        .rename(columns={0: f"poly_tau_{i}"})
        for i in range(5, 41, 5)
    ]
    timed_pocketsizes = pd.concat(
        [
            timed_pocketsizes[0],
            *[tp.drop(columns=["gameId", "playId"]) for tp in timed_pocketsizes[1:]],
        ],
        axis=1,
    )
    print(f"{timed_pocketsizes.shape=}")

    plays_with_collapse = pd.merge(
        mean_polydists,
        plays[
            (plays.defendersInBox.between(4, 8))  ## pff true pass set
            & (plays.down < 4)
            & ~plays.offenseFormation.isin(["JUMBO", "WILDCAT"])
            & ~plays.dropBackType.isin(["DESIGNED_RUN", "UNKNOWN"])
        ][
            [
                "gameId",
                "playId",
                "playDescription",
                "pff_playAction",
                "passResult",
                "prePenaltyPlayResult",
                "defendersInBox",
                "personnelO",
                "personnelD",
                "dropBackType",
                "pff_passCoverage",
                "pff_passCoverageType",
                "offenseFormation",
                "yardsToGo",
                "down",
                "possessionTeam",
                "defensiveTeam",
                "ep",
                "epa",
                "qb_epa",
                "touchdown",
                "air_yards",
            ]
        ],
        how="inner",
        on=["gameId", "playId"],
    )

    plays_with_collapse["complete"] = plays_with_collapse.passResult == "C"
    plays_with_collapse["incomplete"] = plays_with_collapse.passResult == "I"
    plays_with_collapse["sack"] = plays_with_collapse.passResult == "S"

    plays_with_collapse = plays_with_collapse.merge(
        pass_abs_frames, how="left", on=["gameId", "playId"]
    )

    plays_with_collapse = plays_with_collapse.merge(
        min_frames, how="left", on=["gameId", "playId"]
    )

    plays_with_collapse = plays_with_collapse.merge(
        poly_times, how="left", on=["gameId", "playId"]
    )

    plays_with_collapse = plays_with_collapse.merge(
        surv_time, how="left", on=["gameId", "playId"]
    )

    plays_with_collapse = plays_with_collapse.merge(
        pass_frames, how="left", on=["gameId", "playId"]
    )

    plays_with_collapse = plays_with_collapse.merge(
        mean_polydiffs, how="left", on=["gameId", "playId"]
    )

    plays_with_collapse = plays_with_collapse.merge(
        timed_pocketsizes, how="left", on=["gameId", "playId"]
    )

    plays_with_collapse = pd.merge(
        plays_with_collapse,
        qb_spaces[["gameId", "playId", "frameId", "polydist", "qb_radius"]],
        left_on=["gameId", "playId", "pass_absolute_frame"],
        right_on=["gameId", "playId", "frameId"],
        how="left",
    )

    plays_with_collapse = plays_with_collapse.merge(pwin, on=["gameId", "playId"])

    max_frames = (
        qb_spaces.groupby(["gameId", "playId"])
        .frameId.max()
        .reset_index()
        .rename(columns={"frameId": "max_frame"})
    )
    plays_with_collapse = plays_with_collapse.merge(
        max_frames, on=["gameId", "playId"], how="left"
    )

    plays_with_collapse["survived"] = (
        plays_with_collapse.poly_frame == -1
    )  ## it survieved iff no collapse frame
    plays_with_collapse["pass_thrown"] = plays_with_collapse.pass_frame > 0
    plays_with_collapse["pocket_end_event"] = (
        1 - plays_with_collapse["survived"]
    ) | plays_with_collapse["pass_frame"].le(plays_with_collapse["surv_frame"])
    plays_with_collapse["pocket_end"] = plays_with_collapse[
        ["surv_frame", "pass_frame"]
    ].min(axis=1)

    plays_with_collapse["pocket_vol"] = (
        plays_with_collapse["mean_polydists"] * plays_with_collapse["surv_frame"]
    )

    plays_with_collapse["down1"] = plays_with_collapse.down == 1
    plays_with_collapse["down2"] = plays_with_collapse.down == 2
    plays_with_collapse["down3"] = plays_with_collapse.down == 3
    down_cols = ["down2", "down3"]
    plays_with_collapse["zone"] = plays_with_collapse.pff_passCoverageType == "Zone"
    plays_with_collapse["man"] = plays_with_collapse.pff_passCoverageType == "Man"
    mz_cols = ["zone", "man"]
    plays_with_collapse["Cover-3"] = plays_with_collapse.pff_passCoverage == "Cover-3"
    plays_with_collapse["Cover-1"] = plays_with_collapse.pff_passCoverage == "Cover-1"
    plays_with_collapse["Cover-2"] = plays_with_collapse.pff_passCoverage == "Cover-2"
    plays_with_collapse["Quarters"] = plays_with_collapse.pff_passCoverage == "Quarters"
    plays_with_collapse["Cover-6"] = plays_with_collapse.pff_passCoverage == "Cover-6"
    plays_with_collapse["RedZone"] = plays_with_collapse.pff_passCoverage == "Red Zone"
    plays_with_collapse["Cover-0"] = plays_with_collapse.pff_passCoverage == "Cover-0"
    plays_with_collapse["2-Man"] = plays_with_collapse.pff_passCoverage == "2-Man"
    def_cols = [
        "Cover-3",
        "Cover-1",
        "Cover-2",
        "Cover-6",
        "Quarters",
        "Cover-0",
        "2-Man",
        "RedZone",
    ]

    plays_with_collapse["shotgun"] = plays_with_collapse.offenseFormation == "SHOTGUN"
    plays_with_collapse["empty"] = plays_with_collapse.offenseFormation == "EMPTY"
    plays_with_collapse["singleback"] = (
        plays_with_collapse.offenseFormation == "SINGLEBACK"
    )
    plays_with_collapse["iform"] = plays_with_collapse.offenseFormation == "I_FORM"
    off_cols = [
        "shotgun",
        "empty",
        "singleback",
        "iform",
    ]

    plays_with_collapse["pass_frame"] = plays_with_collapse.pass_frame.abs()
    plays_with_collapse["comp_air_yards"] = (
        plays_with_collapse.air_yards * plays_with_collapse.complete
    )
    plays_with_collapse["safe_pass"] = (
        plays_with_collapse.surv_frame > plays_with_collapse.pass_frame
    )

    return plays_with_collapse, off_cols, def_cols
