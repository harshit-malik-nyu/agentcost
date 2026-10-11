import math,pytest
from agentcost.calibration import Answer
from agentcost.conformal import control_risk,clopper_pearson_upper

def test_grid_search_requires_simultaneous_not_pointwise_bound():
 # 0/100 passes a pointwise 5% bound but fails after searching 201 choices.
 rows=[Answer(item_id="fixture",clause_type="fixture",confidence=0.9,correct=True) for _ in range(100)]
 assert clopper_pearson_upper(0,100,.05)<.05
 assert not control_risk(rows,grid=200).feasible

def test_large_zero_error_sample_and_invalid_confidence():
 rows=[Answer(item_id="fixture",clause_type="fixture",confidence=.9,correct=True) for _ in range(1000)]
 p=control_risk(rows)
 assert p.feasible and p.threshold==0
 assert p.risk_upper_bound==pytest.approx(1-(.05/201)**(1/1000))
 with pytest.raises(ValueError):control_risk([Answer(item_id="fixture",clause_type="fixture",confidence=math.nan,correct=True)])
