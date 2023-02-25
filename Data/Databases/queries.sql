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



SELECT trial_num, AVG(winnings) AvgWinnings, MIN(winnings) MinWinnings, MAX(winnings) MaxWinnings
FROM (
		SELECT trial_num, 
			   repeat_num, 
			   sum(CASE WHEN avg_winnings > 25000 THEN 25000 ELSE avg_winnings END) winnings 
			   FROM Entry_Optimize_Results
		WHERE trial_num > 161
		GROUP BY trial_num, repeat_num
)
GROUP BY trial_num
ORDER BY AVG(winnings) DESC;



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