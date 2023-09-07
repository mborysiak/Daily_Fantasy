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



SELECT trial_num, 
	   AVG(winnings) AvgWinnings, 
	   MIN(winnings) MinWinnings, 
	   MAX(winnings) MaxWinnings, 
	   AVG(non8_winnings) AvgNon8Winnings, 
	   MIN(non8_winnings) MinNon8Winnings,
	   MAX(non8_winnings) MaxNon8Winnings,
	   (AVG(winnings)+AVG(non8_winnings)+MIN(winnings)+MIN(non8_winnings))/4 as BlendedAverage
FROM (
		SELECT trial_num, 
			   repeat_num, 
			   sum(CASE WHEN avg_winnings > 10000 THEN 10000 ELSE avg_winnings END) winnings,
			   sum(CASE WHEN week=8 THEN 0 
			            WHEN avg_winnings > 10000 THEN 10000 
						ELSE avg_winnings END) non8_winnings 
		FROM Entry_Optimize_Results
		WHERE trial_num >= 269
		      AND week < 17
		GROUP BY trial_num, repeat_num
)
GROUP BY trial_num
ORDER BY AVG(winnings)+AVG(non8_winnings)+MIN(winnings)+MIN(non8_winnings) DESC;



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


SELECT DISTINCT version, ensemble_vers, std_dev_type 
FROM Model_Predictions 
WHERE week=3 and year=2022;