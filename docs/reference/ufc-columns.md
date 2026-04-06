# UFC Dataset Column Tracker

6528 rows, 118 columns. Mark columns KEEP or DROP.

## Target


| #   | Column | Status | Type | Nulls | Notes                               |
| --- | ------ | ------ | ---- | ----- | ----------------------------------- |
| 9   | Winner | KEEP   | str  | 0     | "Red" or "Blue" — response variable |


## Fight Metadata


| #   | Column         | Status | Type  | Nulls | Notes                |
| --- | -------------- | ------ | ----- | ----- | -------------------- |
| 0   | RedFighter     | KEEP   | str   | 0     | Fighter name         |
| 1   | BlueFighter    | KEEP   | str   | 0     | Fighter name         |
| 6   | Date           | KEEP   | str   | 0     | e.g. 2024-12-07      |
| 7   | Location       | KEEP   | str   | 0     | City, State, Country |
| 8   | Country        | KEEP   | str   | 0     |                      |
| 10  | TitleBout      | KEEP   | bool  | 0     |                      |
| 11  | WeightClass    | KEEP   | str   | 0     | e.g. Flyweight       |
| 12  | Gender         | KEEP   | str   | 0     | MALE/FEMALE          |
| 13  | NumberOfRounds | KEEP   | int   | 0     | 3 or 5               |
| 77  | EmptyArena     | KEEP   | float | 1486  | COVID era flag       |


## Betting Odds


| #   | Column            | Status | Type  | Nulls | Notes                          |
| --- | ----------------- | ------ | ----- | ----- | ------------------------------ |
| 2   | RedOdds           | DROP   | float | 227   | replaced by derived cols       |
| 3   | BlueOdds          | DROP   | float | 226   | replaced by derived cols       |
| 4   | RedExpectedValue  | DROP   | float | 227   | redundant with odds            |
| 5   | BlueExpectedValue | DROP   | float | 226   | redundant with odds            |
| 112 | RedDecOdds        | DROP   | float | 1087  | method-specific, not wanted    |
| 113 | BlueDecOdds       | DROP   | float | 1116  | method-specific, not wanted    |
| 114 | RSubOdds          | DROP   | float | 1336  | method-specific, not wanted    |
| 115 | BSubOdds          | DROP   | float | 1359  | method-specific, not wanted    |
| 116 | RKOOdds           | DROP   | float | 1334  | method-specific, not wanted    |
| 117 | BKOOdds           | DROP   | float | 1360  | method-specific, not wanted    |
| NEW | RedDecimalOdds    | KEEP   | float | 227   | derived: decimal multiplier    |
| NEW | BlueDecimalOdds   | KEEP   | float | 226   | derived: decimal multiplier    |
| NEW | RedImpliedProb    | KEEP   | float | 227   | derived: normalized impl. prob |
| NEW | BlueImpliedProb   | KEEP   | float | 226   | derived: normalized impl. prob |


## Red Fighter — Record & Streaks


| #   | Column               | Status | Type | Nulls | Notes |
| --- | -------------------- | ------ | ---- | ----- | ----- |
| 37  | RedCurrentLoseStreak | KEEP   | int  | 0     |       |
| 38  | RedCurrentWinStreak  | KEEP   | int  | 0     |       |
| 39  | RedDraws             | KEEP   | int  | 0     |       |
| 45  | RedLongestWinStreak  | KEEP   | int  | 0     |       |
| 46  | RedLosses            | KEEP   | int  | 0     |       |
| 55  | RedWins              | KEEP   | int  | 0     |       |
| 47  | RedTotalRoundsFought | KEEP   | int  | 0     |       |
| 48  | RedTotalTitleBouts   | KEEP   | int  | 0     |       |


## Red Fighter — Win Breakdown


| #   | Column                     | Status | Type | Nulls | Notes |
| --- | -------------------------- | ------ | ---- | ----- | ----- |
| 49  | RedWinsByDecisionMajority  | DROP   | int  | 0     |       |
| 50  | RedWinsByDecisionSplit     | DROP   | int  | 0     |       |
| 51  | RedWinsByDecisionUnanimous | DROP   | int  | 0     |       |
| 52  | RedWinsByKO                | DROP   | int  | 0     |       |
| 53  | RedWinsBySubmission        | DROP   | int  | 0     |       |
| 54  | RedWinsByTKODoctorStoppage | DROP   | int  | 0     |       |


## Red Fighter — Fight Stats (Averages)


| #   | Column             | Status | Type  | Nulls | Notes                  |
| --- | ------------------ | ------ | ----- | ----- | ---------------------- |
| 40  | RedAvgSigStrLanded | KEEP   | float | 455   | Sig. strikes per fight |
| 41  | RedAvgSigStrPct    | KEEP   | float | 357   | Sig. strike accuracy   |
| 42  | RedAvgSubAtt       | KEEP   | float | 357   | Sub attempts per fight |
| 43  | RedAvgTDLanded     | KEEP   | float | 357   | Takedowns per fight    |
| 44  | RedAvgTDPct        | KEEP   | float | 367   | Takedown accuracy      |


## Red Fighter — Physical


| #   | Column       | Status | Type  | Nulls | Notes                    |
| --- | ------------ | ------ | ----- | ----- | ------------------------ |
| 56  | RedStance    | KEEP   | str   | 0     | Orthodox/Southpaw/Switch |
| 57  | RedHeightCms | KEEP   | float | 0     |                          |
| 58  | RedReachCms  | KEEP   | float | 0     |                          |
| 59  | RedWeightLbs | KEEP   | int   | 0     |                          |
| 60  | RedAge       | KEEP   | int   | 0     |                          |


## Blue Fighter — Record & Streaks


| #   | Column                | Status | Type | Nulls | Notes |
| --- | --------------------- | ------ | ---- | ----- | ----- |
| 14  | BlueCurrentLoseStreak | KEEP   | int  | 0     |       |
| 15  | BlueCurrentWinStreak  | KEEP   | int  | 0     |       |
| 16  | BlueDraws             | KEEP   | int  | 0     |       |
| 22  | BlueLongestWinStreak  | KEEP   | int  | 0     |       |
| 23  | BlueLosses            | KEEP   | int  | 0     |       |
| 32  | BlueWins              | KEEP   | int  | 0     |       |
| 24  | BlueTotalRoundsFought | KEEP   | int  | 0     |       |
| 25  | BlueTotalTitleBouts   | KEEP   | int  | 0     |       |


## Blue Fighter — Win Breakdown


| #   | Column                      | Status | Type | Nulls | Notes |
| --- | --------------------------- | ------ | ---- | ----- | ----- |
| 26  | BlueWinsByDecisionMajority  | DROP   | int  | 0     |       |
| 27  | BlueWinsByDecisionSplit     | DROP   | int  | 0     |       |
| 28  | BlueWinsByDecisionUnanimous | DROP   | int  | 0     |       |
| 29  | BlueWinsByKO                | DROP   | int  | 0     |       |
| 30  | BlueWinsBySubmission        | DROP   | int  | 0     |       |
| 31  | BlueWinsByTKODoctorStoppage | DROP   | int  | 0     |       |


## Blue Fighter — Fight Stats (Averages)


| #   | Column              | Status | Type  | Nulls | Notes                  |
| --- | ------------------- | ------ | ----- | ----- | ---------------------- |
| 17  | BlueAvgSigStrLanded | KEEP   | float | 930   | Sig. strikes per fight |
| 18  | BlueAvgSigStrPct    | KEEP   | float | 765   | Sig. strike accuracy   |
| 19  | BlueAvgSubAtt       | KEEP   | float | 832   | Sub attempts per fight |
| 20  | BlueAvgTDLanded     | KEEP   | float | 833   | Takedowns per fight    |
| 21  | BlueAvgTDPct        | KEEP   | float | 842   | Takedown accuracy      |


## Blue Fighter — Physical


| #   | Column        | Status | Type  | Nulls | Notes                    |
| --- | ------------- | ------ | ----- | ----- | ------------------------ |
| 33  | BlueStance    | KEEP   | str   | 3     | Orthodox/Southpaw/Switch |
| 34  | BlueHeightCms | KEEP   | float | 0     |                          |
| 35  | BlueReachCms  | KEEP   | float | 0     |                          |
| 36  | BlueWeightLbs | KEEP   | int   | 0     |                          |
| 61  | BlueAge       | KEEP   | int   | 0     |                          |


## Difference Features (Blue - Red)


| #   | Column              | Status | Type  | Nulls | Notes   |
| --- | ------------------- | ------ | ----- | ----- | ------- |
| 62  | LoseStreakDif       | DROP   | int   | 0     | derived |
| 63  | WinStreakDif        | DROP   | int   | 0     | derived |
| 64  | LongestWinStreakDif | DROP   | int   | 0     | derived |
| 65  | WinDif              | DROP   | int   | 0     | derived |
| 66  | LossDif             | DROP   | int   | 0     | derived |
| 67  | TotalRoundDif       | DROP   | int   | 0     | derived |
| 68  | TotalTitleBoutDif   | DROP   | int   | 0     | derived |
| 69  | KODif               | DROP   | int   | 0     | derived |
| 70  | SubDif              | DROP   | int   | 0     | derived |
| 71  | HeightDif           | DROP   | float | 0     | derived |
| 72  | ReachDif            | DROP   | float | 0     | derived |
| 73  | AgeDif              | DROP   | int   | 0     | derived |
| 74  | SigStrDif           | DROP   | float | 0     | derived |
| 75  | AvgSubAttDif        | DROP   | float | 0     | derived |
| 76  | AvgTDDif            | DROP   | float | 0     | derived |


## Rankings (per weight class)


| #   | Column       | Status | Type  | Nulls | Notes                   |
| --- | ------------ | ------ | ----- | ----- | ----------------------- |
| 79  | RMatchWCRank | DROP   | float | 4749  | sparse |
| 78  | BMatchWCRank | DROP   | float | 5328  | sparse |
| 92  | RPFPRank     | DROP   | float | 6275  | sparse |
| 105 | BPFPRank     | DROP   | float | 6461  | sparse |
| 106 | BetterRank   | DROP   | str   | 0     | derived from ranks |


### Red per-division ranks (very sparse, ~96-99% null)


| #   | Column                | Status | Type  | Nulls | Notes |
| --- | --------------------- | ------ | ----- | ----- | ----- |
| 80  | RWFlyweightRank       | DROP   | float | 6432  | sparse |
| 81  | RWFeatherweightRank   | DROP   | float | 6519  | sparse |
| 82  | RWStrawweightRank     | DROP   | float | 6382  | sparse |
| 83  | RWBantamweightRank    | DROP   | float | 6374  | sparse |
| 84  | RHeavyweightRank      | DROP   | float | 6342  | sparse |
| 85  | RLightHeavyweightRank | DROP   | float | 6344  | sparse |
| 86  | RMiddleweightRank     | DROP   | float | 6346  | sparse |
| 87  | RWelterweightRank     | DROP   | float | 6337  | sparse |
| 88  | RLightweightRank      | DROP   | float | 6344  | sparse |
| 89  | RFeatherweightRank    | DROP   | float | 6351  | sparse |
| 90  | RBantamweightRank     | DROP   | float | 6347  | sparse |
| 91  | RFlyweightRank        | DROP   | float | 6340  | sparse |


### Blue per-division ranks (very sparse, ~96-99% null)


| #   | Column                | Status | Type  | Nulls | Notes |
| --- | --------------------- | ------ | ----- | ----- | ----- |
| 93  | BWFlyweightRank       | DROP   | float | 6455  | sparse |
| 94  | BWFeatherweightRank   | DROP   | float | 6527  | sparse |
| 95  | BWStrawweightRank     | DROP   | float | 6428  | sparse |
| 96  | BWBantamweightRank    | DROP   | float | 6421  | sparse |
| 97  | BHeavyweightRank      | DROP   | float | 6380  | sparse |
| 98  | BLightHeavyweightRank | DROP   | float | 6408  | sparse |
| 99  | BMiddleweightRank     | DROP   | float | 6391  | sparse |
| 100 | BWelterweightRank     | DROP   | float | 6409  | sparse |
| 101 | BLightweightRank      | DROP   | float | 6408  | sparse |
| 102 | BFeatherweightRank    | DROP   | float | 6404  | sparse |
| 103 | BBantamweightRank     | DROP   | float | 6409  | sparse |
| 104 | BFlyweightRank        | DROP   | float | 6398  | sparse |


## Fight Outcome Details (post-fight — potential leakage)


| #   | Column             | Status | Type  | Nulls | Notes                   |
| --- | ------------------ | ------ | ----- | ----- | ----------------------- |
| 107 | Finish             | DROP   | str   | 238   | leakage |
| 108 | FinishDetails      | DROP   | str   | 3636  | leakage |
| 109 | FinishRound        | DROP   | float | 622   | leakage |
| 110 | FinishRoundTime    | DROP   | str   | 622   | leakage |
| 111 | TotalFightTimeSecs | DROP   | float | 622   | leakage |


