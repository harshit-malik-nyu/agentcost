"""Conditional review-time boundaries; no measured ROI or validated risk claim."""
from pathlib import Path
from decimal import Decimal as D
from copy import deepcopy
import hashlib,json
from decision_lab import economics
HERE=Path(__file__).resolve().parent

def calculate(spec,share=D('0.35')):
    e=spec['economics'];manual,triage=e['options'][:2]
    assert D('0')<share<=D('1')
    v=D(str(e['volume']));capacity=D(str(e['review_capacity_hours']));loss=D(str(e['loss_per_incident_usd']))
    # Assert the selected driver by stable activity name rather than quietly shifting scope.
    driver=next(a for a in triage['activities'] if a['name']=='Exception review: 35% at 9 minutes')
    rate=D(str(driver['usd_per_unit']));assert v>0 and rate>0
    manual_variable=sum(D(str(a['units_per_case']))*D(str(a['usd_per_unit'])) for a in manual['activities'])+D(str(manual['incident_rate']))*loss
    manual_total=D(str(manual['fixed_usd']))+v*manual_variable
    other=[a for a in triage['activities'] if a is not driver]
    other_hours=sum(D(str(a['units_per_case'])) for a in other if a['unit']=='reviewer_hour')
    other_cost=D(str(triage['fixed_usd']))+v*(sum(D(str(a['units_per_case']))*D(str(a['usd_per_unit'])) for a in other)+D(str(triage['incident_rate']))*loss)
    cost_minutes=(manual_total-other_cost)/(v*share*rate)*60
    capacity_minutes=(capacity/v-other_hours)/share*60
    return {'costTieMinutes':float(cost_minutes),'capacityLimitMinutes':float(capacity_minutes),
      'reviewShare':float(share),'manualEconomicCost':float(manual_total),
      'manualHours':float(v*sum(D(str(a['units_per_case'])) for a in manual['activities'] if a['unit']=='reviewer_hour'))}

def at_minutes(spec,minutes):
    assert minutes>=0
    e=deepcopy(spec['economics'])
    activity=next(a for a in e['options'][1]['activities'] if a['name']=='Exception review: 35% at 9 minutes')
    activity['units_per_case']=float(D('0.35')*D(str(minutes))/60)
    activity['name']=f'Exception review: 35% at {minutes} minutes'
    result=economics(e)
    result['exceptionReviewMinutes']=minutes
    return result

def build():
    spec=json.loads((HERE/'decision_spec.json').read_text());bounds=calculate(spec)
    rows=[at_minutes(spec,m) for m in [9,18,19,20]]
    obj={'evidence':'Illustrative planning assumptions only. No measured review time, incident risk or cash savings.',
      'scope':'One-way review-time sensitivity at fixed 35% exception share, volume, costs, QA and assumed risk. Does not optimize or validate the relationship between review effort and safety.',
      'boundaries':bounds,'scenarios':rows,
      'sourceSha256':{p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in ['decision_spec.json','decision_lab.py','review_boundary.py','test_review_boundary.py']}}
    out=HERE/'output/review_boundary';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    lines=['# When does extra review reverse the decision?','',obj['evidence'],'',
      f"At the fixed 35% review share, triage ties the theoretical review-all economic cost at **{bounds['costTieMinutes']:.2f} minutes per exception**. It reaches the 2,400-hour capacity limit at **{bounds['capacityLimitMinutes']:.2f} minutes**.",'',
      '**Important:** review-all already needs 2,666.7 hours at the assumed 20,000-case volume. It is infeasible at the declared capacity. Crossing its cost line is not permission to recommend it. At 19 minutes triage remains the only feasible option in this planning model; at 20 minutes no option is feasible.','',
      '| Exception minutes | Triage economic cost | Cash outlay | Review hours | Model selection |','|---:|---:|---:|---:|---|']
    for r in rows:
        t=r['options'][1]
        lines.append(f"| {r['exceptionReviewMinutes']} | ${t['economic_cost_usd']:,.2f} | ${t['cash_outlay_usd']:,.2f} | {t['reviewer_hours']:,.1f} | {r['recommended_if_assumptions_hold'] or 'NO FEASIBLE OPTION'} |")
    lines+=['',obj['scope'],'','## Decision and reversal conditions','',
      '- Base planning case: request a bounded evaluation; do not authorize deployment from assumed incident rates.',
      '- If review time rises, update both the activity cost and workload. Do not compare only model API spending.',
      '- If no option passes capacity and risk, hold or revise the operating model with explicit new assumptions; do not choose the least expensive failing option.',
      '- Salaried review hours are valued capacity, not automatically avoidable cash expenditure. Releasing time is not equivalent to a headcount saving.',
      '- The actual risk bound, workload mix and review-time distribution must be measured separately; averages and annual capacity do not establish peak staffing or queue SLAs.','',
      '## Reproduce','', '`python analysis/test_review_boundary.py` then `python analysis/review_boundary.py`. Python standard library only. The tests compare Decimal-derived boundaries with the existing cost engine on both sides of each boundary and retain risk/capacity vetoes.']
    (out/'decision.md').write_text('\n'.join(lines)+'\n');print(json.dumps(bounds))
if __name__=='__main__':build()
