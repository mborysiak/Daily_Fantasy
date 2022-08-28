SELECT week, pred_vers, ensemble_vers, std_dev_type, sim_type, AVG(max_winnings), AVG(total_winnings), MAX(max_winnings), MAX(total_winnings)
FROM Winnings_Optimize
WHERE week=9
GROUP BY week, pred_vers, ensemble_vers, std_dev_type, sim_type
ORDER BY AVG(max_winnings) DESC