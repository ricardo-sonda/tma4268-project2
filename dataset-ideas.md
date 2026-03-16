# Dataset ideas

| key | dataset | kaggle id | note |
| --- | --- | --- | --- |
| `ufc` | UFC Dataset 1994-2026 | `jossilva3110/ufc-dataset-1994-2026` | classification idea |
| `fifa24` | FIFA 24 Player Stats | `rehandl23/fifa-24-player-stats-dataset` | regression idea |

## UFC columns organized

### How to read the naming
- `Red*` and `Blue*`: fighter-specific variables for the two corners.
- `*Dif`: engineered difference features, usually `Red - Blue`.
- `R*Rank` and `B*Rank`: ranking fields for red/blue fighters.
- Some columns are pre-fight inputs, while others are post-fight outcomes and should not be used as predictors if you are building a model.

### 1. Fight and event context
- Fighters: `RedFighter`, `BlueFighter`
- Event info: `Date`, `Location`, `Country`
- Bout setup: `TitleBout`, `WeightClass`, `Gender`, `NumberOfRounds`, `EmptyArena`

### 2. Outcome columns
- Winner/outcome: `Winner`
- Finish info: `Finish`, `FinishDetails`, `FinishRound`, `FinishRoundTime`, `TotalFightTimeSecs`

### 3. Betting market columns
- Main odds: `RedOdds`, `BlueOdds`
- Expected value: `RedExpectedValue`, `BlueExpectedValue`
- Method-specific odds: `RedDecOdds`, `BlueDecOdds`, `RSubOdds`, `BSubOdds`, `RKOOdds`, `BKOOdds`

### 4. Blue fighter features
- Form and record: `BlueCurrentLoseStreak`, `BlueCurrentWinStreak`, `BlueDraws`, `BlueLongestWinStreak`, `BlueLosses`, `BlueWins`
- Striking/grappling averages: `BlueAvgSigStrLanded`, `BlueAvgSigStrPct`, `BlueAvgSubAtt`, `BlueAvgTDLanded`, `BlueAvgTDPct`
- Experience: `BlueTotalRoundsFought`, `BlueTotalTitleBouts`
- Win breakdown: `BlueWinsByDecisionMajority`, `BlueWinsByDecisionSplit`, `BlueWinsByDecisionUnanimous`, `BlueWinsByKO`, `BlueWinsBySubmission`, `BlueWinsByTKODoctorStoppage`
- Physical profile: `BlueStance`, `BlueHeightCms`, `BlueReachCms`, `BlueWeightLbs`, `BlueAge`

### 5. Red fighter features
- Form and record: `RedCurrentLoseStreak`, `RedCurrentWinStreak`, `RedDraws`, `RedLongestWinStreak`, `RedLosses`, `RedWins`
- Striking/grappling averages: `RedAvgSigStrLanded`, `RedAvgSigStrPct`, `RedAvgSubAtt`, `RedAvgTDLanded`, `RedAvgTDPct`
- Experience: `RedTotalRoundsFought`, `RedTotalTitleBouts`
- Win breakdown: `RedWinsByDecisionMajority`, `RedWinsByDecisionSplit`, `RedWinsByDecisionUnanimous`, `RedWinsByKO`, `RedWinsBySubmission`, `RedWinsByTKODoctorStoppage`
- Physical profile: `RedStance`, `RedHeightCms`, `RedReachCms`, `RedWeightLbs`, `RedAge`

### 6. Engineered comparison features
- Form/record differences: `LoseStreakDif`, `WinStreakDif`, `LongestWinStreakDif`, `WinDif`, `LossDif`
- Experience differences: `TotalRoundDif`, `TotalTitleBoutDif`
- Style/finish differences: `KODif`, `SubDif`, `SigStrDif`, `AvgSubAttDif`, `AvgTDDif`
- Physical differences: `HeightDif`, `ReachDif`, `AgeDif`

### 7. Ranking columns
- Matchup ranking summary: `BMatchWCRank`, `RMatchWCRank`, `BetterRank`
- Red fighter rankings: `RWFlyweightRank`, `RWFeatherweightRank`, `RWStrawweightRank`, `RWBantamweightRank`, `RHeavyweightRank`, `RLightHeavyweightRank`, `RMiddleweightRank`, `RWelterweightRank`, `RLightweightRank`, `RFeatherweightRank`, `RBantamweightRank`, `RFlyweightRank`, `RPFPRank`
- Blue fighter rankings: `BWFlyweightRank`, `BWFeatherweightRank`, `BWStrawweightRank`, `BWBantamweightRank`, `BHeavyweightRank`, `BLightHeavyweightRank`, `BMiddleweightRank`, `BWelterweightRank`, `BLightweightRank`, `BFeatherweightRank`, `BBantamweightRank`, `BFlyweightRank`, `BPFPRank`

### 8. Most useful high-level interpretation
- `target` candidate: `Winner`
- Strong pre-fight predictors: odds, fighter records, striking/grappling averages, age, height, reach, rankings
- Likely leakage for prediction: `Finish`, `FinishDetails`, `FinishRound`, `FinishRoundTime`, `TotalFightTimeSecs`
