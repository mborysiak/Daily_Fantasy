SELECT week, pred_vers, ensemble_vers, std_dev_type, AVG(max_winnings)
FROM Winnings_Optimize
GROUP BY week, pred_vers, ensemble_vers, std_dev_type
ORDER BY AVG(max_winnings) DESC