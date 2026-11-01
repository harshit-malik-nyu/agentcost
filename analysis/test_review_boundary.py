from pathlib import Path
import json,unittest
from copy import deepcopy
from review_boundary import calculate,at_minutes
S=json.loads(Path(__file__).with_name('decision_spec.json').read_text())
class Boundaries(unittest.TestCase):
    def test_base_reconciles_existing_model(self):
        r=at_minutes(S,9);t=r['options'][1]
        self.assertEqual(t['economic_cost_usd'],163100);self.assertEqual(t['cash_outlay_usd'],19600);self.assertEqual(t['reviewer_hours'],1150)
    def test_cost_boundary_does_not_approve_infeasible_baseline(self):
        b=calculate(S)
        for sign in [-1,1]:
            r=at_minutes(S,b['costTieMinutes']+sign*.01);a,t=r['options'][:2]
            self.assertEqual(t['economic_cost_usd']>a['economic_cost_usd'],sign==1)
            self.assertFalse(a['feasible']);self.assertEqual(r['recommended_if_assumptions_hold'],t['option'])
    def test_capacity_boundary_blocks_every_option(self):
        b=calculate(S);before=at_minutes(S,b['capacityLimitMinutes']-.01);after=at_minutes(S,b['capacityLimitMinutes']+.01)
        self.assertTrue(before['options'][1]['capacity_gate_pass']);self.assertFalse(after['options'][1]['capacity_gate_pass']);self.assertIsNone(after['recommended_if_assumptions_hold'])
    def test_cheap_review_cannot_bypass_risk(self):
        s=deepcopy(S);s['economics']['options'][1]['incident_rate']=.02
        self.assertIsNone(at_minutes(s,0)['recommended_if_assumptions_hold'])
    def test_cash_stays_fixed_while_valued_capacity_changes(self):
        a,b=at_minutes(S,9)['options'][1],at_minutes(S,19)['options'][1]
        self.assertEqual(a['cash_outlay_usd'],b['cash_outlay_usd']);self.assertGreater(b['economic_cost_usd'],a['economic_cost_usd'])
if __name__=='__main__':unittest.main()
