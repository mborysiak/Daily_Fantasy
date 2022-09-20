SELECT week, year, pred_vers, ensemble_vers, std_dev_type,  NumPlayers, sim_type, Contest,
		AVG(max_winnings), AVG(total_winnings), MAX(max_winnings), MAX(total_winnings)
FROM Winnings_Optimize
WHERE week=17
      AND year = 2021
GROUP BY week, pred_vers, ensemble_vers, std_dev_type,  NumPlayers, sim_type, Contest
ORDER BY AVG(max_winnings) DESC