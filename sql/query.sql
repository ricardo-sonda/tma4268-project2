SELECT RedDecimalOdds, BlueDecimalOdds
FROM interim__ultimate_ufc__ufc_clean;

SELECT
    BlueFighter,
    RedFighter,
    Date,
    (RedDecimalOdds * RedImpliedProb + BlueDecimalOdds * BlueImpliedProb) / 2 AS payout
FROM interim__ultimate_ufc__ufc_clean;
