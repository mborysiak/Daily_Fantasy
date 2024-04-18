SELECT week, year, pred_vers, ensemble_vers, std_dev_type,  NumPlayers, sim_type, Contest,
		AVG(max_winnings), AVG(total_winnings), MAX(max_winnings), MAX(total_winnings), AVG(max_points), AVG(avg_points)
FROM Winnings_Optimize
WHERE week=5
      AND year = 2022
GROUP BY week, pred_vers, ensemble_vers, std_dev_type,  NumPlayers, sim_type, Contest
ORDER BY AVG(max_winnings) DESC;



SELECT week, year, trial_num, repeat_num, lineup_num, unique_lineup_num, sum(fantasy_pts) total_points
FROM Entry_Optimize_Lineups
GROUP BY week, year, trial_num, repeat_num, lineup_num, unique_lineup_num
ORDER BY sum(fantasy_pts) DESC;


SELECT * 
FROM (
SELECT reg_ens_vers,
	   million_ens_vers,
	   std_dev_type,
	   trial_num, 
	   AVG(winnings) AvgWinnings, 
	   MIN(winnings) MinWinnings, 
	   MAX(winnings) MaxWinnings, 
	   AVG(non8_winnings) AvgNon8Winnings, 
	   MIN(non8_winnings) MinNon8Winnings,
	   MAX(non8_winnings) MaxNon8Winnings,
	   ROUND((AVG(winnings)+AVG(non8_winnings)+MIN(winnings)+MIN(non8_winnings))/4,1) as BlendedAvgMin,
	   ROUND((AVG(winnings)+AVG(non8_winnings)+MIN(winnings)+MIN(non8_winnings)+(MAX(winnings)/2)+(MAX(non8_winnings)/2))/6,1) as BlendedAvgMinMax
FROM (
		SELECT reg_ens_vers,
		       million_ens_vers,
			   std_dev_type,
			   trial_num, 
			   repeat_num, 
			   sum(CASE WHEN avg_winnings > 10000 THEN 10000 ELSE avg_winnings END) winnings,
			   sum(CASE WHEN week=8 and year=2022 THEN 0 
			            WHEN avg_winnings > 10000 THEN 10000 
						ELSE avg_winnings END) non8_winnings 
		FROM Entry_Optimize_Results
		WHERE trial_num >= 460
		      AND week < 17
		GROUP BY trial_num, repeat_num
)
GROUP BY trial_num,
	     reg_ens_vers,
	     million_ens_vers,
	     std_dev_type
)
ORDER BY BlendedAvgMinMax DESC;



WITH all_winnings_tbl AS (

		SELECT reg_ens_vers,
			   million_ens_vers,
			   std_dev_type,
			   trial_num, 
			   AVG(winnings) AvgWinnings, 
			   MIN(winnings) MinWinnings, 
			   MAX(winnings) MaxWinnings
		FROM (
				SELECT reg_ens_vers,
					   million_ens_vers,
					   std_dev_type,
					   trial_num, 
					   repeat_num, 
					   sum(CASE WHEN avg_winnings > 10000 THEN 10000 ELSE avg_winnings END) winnings,
					   row_number() OVER (PARTITION BY trial_num ORDER BY sum(CASE WHEN avg_winnings > 10000 THEN 10000 ELSE avg_winnings END) DESC) rn_with
				FROM Entry_Optimize_Results
				WHERE trial_num >= 520
					  AND week < 17
				GROUP BY trial_num, repeat_num
		)
		WHERE rn_with > 2 AND rn_with < 8
		GROUP BY trial_num,
				 reg_ens_vers,
				 million_ens_vers,
				 std_dev_type

),
non8_winnings_tbl 
AS (
		SELECT trial_num, 
			   AVG(non8_winnings) AvgNon8Winnings, 
			   MIN(non8_winnings) MinNon8Winnings,
			   MAX(non8_winnings) MaxNon8Winnings
		FROM (
				SELECT trial_num, 
					   repeat_num, 
					   sum(CASE WHEN week=8 and year=2022 THEN 0 
								WHEN avg_winnings > 10000 THEN 10000 
								ELSE avg_winnings END) non8_winnings ,
					   row_number() OVER (PARTITION BY trial_num ORDER BY sum(CASE WHEN week=8 and year=2022 THEN 0 
																				   WHEN avg_winnings > 10000 THEN 10000 
																				   ELSE avg_winnings END) DESC) rn_non8
				FROM Entry_Optimize_Results
				WHERE trial_num >= 520
					  AND week < 17
				GROUP BY trial_num, repeat_num
			)
		WHERE rn_non8>2 AND rn_non8<8
		GROUP BY trial_num
)
SELECT *,
	   ROUND((AvgWinnings+AvgNon8Winnings+MinWinnings+MinNon8Winnings)/4,1) as BlendedAvgMin,
	   ROUND((AvgWinnings+AvgNon8Winnings+MinWinnings+MinNon8Winnings+MaxNon8Winnings+MaxWinnings)/6,1) as BlendedAvgMinMax
FROM all_winnings_tbl
JOIN (
	  SELECT * 
	  FROM non8_winnings_tbl
	  ) USING (trial_num)
ORDER BY BlendedAvgMin DESC



SELECT reg_ens_vers,
		       million_ens_vers,
			   std_dev_type,
			   trial_num, 
			   repeat_num, 
			   sum(CASE WHEN avg_winnings > 10000 THEN 10000 ELSE avg_winnings END) winnings,
			   sum(CASE WHEN week=8 and year=2022 THEN 0 
			            WHEN avg_winnings > 10000 THEN 10000 
						ELSE avg_winnings END) non8_winnings,
			   row_number() OVER (PARTITION BY trial_num ORDER BY sum(CASE WHEN avg_winnings > 10000 THEN 10000 ELSE avg_winnings END) DESC) rn_with,	
			   row_number() OVER (PARTITION BY trial_num ORDER BY sum(CASE WHEN week=8 and year=2022 THEN 0 
																			WHEN avg_winnings > 10000 THEN 10000 
																			ELSE avg_winnings END) DESC) rn_non8
		FROM Entry_Optimize_Results
		WHERE trial_num >= 269
		      AND week < 17
		GROUP BY trial_num, repeat_num;

SELECT trial_num, SUM(WeekTrialRank) as AvgRank
FROM (
		SELECT week, 
			   trial_num,
			   RANK() OVER (PARTITION BY week ORDER BY AvgWinnings DESC) as WeekTrialRank
		FROM (   
				SELECT week,
					   trial_num, 
					   AVG(winnings) AvgWinnings, 
					   MIN(winnings) MinWinnings, 
					   MAX(winnings) MaxWinnings
				FROM (
						SELECT week,
							   trial_num, 
							   repeat_num, 
							   sum(CASE WHEN avg_winnings > 5000 THEN 5000 ELSE avg_winnings END) winnings 
						FROM Entry_Optimize_Results
						WHERE trial_num >= 161
						GROUP BY week, trial_num, repeat_num
				)
				GROUP BY week, trial_num
		)
)
GROUP BY trial_num
ORDER BY SUM(WeekTrialRank) ASC;


SELECT DISTINCT reg_ens_vers, std_dev_type, week, year
FROM Model_Predictions 
WHERE pred_vers='sera0_rsq0_mse1_brier1_matt1_bayes'
     and reg_ens_vers = 'random_kbest_sera0_rsq0_mse1_include2_kfold3'
ORDER BY year, week;

SELECT DISTINCT reg_ens_vers, covar_type, week, year
FROM Covar_Means 
WHERE pred_vers='sera0_rsq0_mse1_brier1_matt1_bayes'
   --  and reg_ens_vers = 'random_kbest_sera0_rsq0_mse1_include2_kfold3'
ORDER BY reg_ens_vers, covar_type, year, week;


SELECT DISTINCT million_ens_vers,  week, year
FROM Predicted_Million 
WHERE pred_vers='sera0_rsq0_mse1_brier1_matt1_bayes'
     and million_ens_vers = 'random_full_stack_matt0_brier1_include2_kfold3'
ORDER BY year, week;