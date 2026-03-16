SELECT RedDecimalOdds, BlueDecimalOdds FROM ufc_clean; 
SELECT BlueFighter, RedFighter, Date, (RedDecimalOdds * RedImpliedProb + BlueDecimalOdds * BlueImpliedProb)/2 AS payout FROM ufc_clean;
