SELECT model,COUNT(*) cases,COUNT(DISTINCT contract) contracts,AVG(correct) accuracy,AVG(confidence) mean_confidence FROM observations GROUP BY model ORDER BY model;
SELECT model,contract,COUNT(*) clauses,AVG(correct) accuracy,RANK() OVER(PARTITION BY model ORDER BY AVG(correct)) review_priority FROM observations GROUP BY model,contract ORDER BY model,review_priority,contract;
