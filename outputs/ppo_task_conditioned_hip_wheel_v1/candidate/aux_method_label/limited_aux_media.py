"""Candidate-only explicit PPO + LIMITED AUX media; no import-time encoding."""
from __future__ import annotations
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from limited_aux_provenance import validate_auxiliary_checkpoint

OUT=Path(__file__).resolve().parents[2]
ROOT=OUT.parents[1]
REVIEW_HASH='1a7e718de692f73c5aa683037474126e2bb7efc39683c864bbe12dab90d5f20b'
QUANTITY_HASH='83139cb604bf6c3beeb826e4ff2d5deb1bba9b55cb2920529ae9ec03e737c7c1'
OLD_SCHEMA='wlr50_clean.task_conditioned_hip_wheel_review.v1'
SCHEMA='wlr50_clean.task_conditioned_limited_aux_review.v1'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


r=module('aux_original_review',OUT/'review_video.py')
require=r.require


def function(path,name,expected_hash):
    require(r.sha256(path)==expected_hash,'reviewed media implementation changed; explicit re-review required')
    return copy.deepcopy(next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
        if isinstance(n,ast.FunctionDef) and n.name==name))


def compiled(node,namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        '<strictly-bound-aux-media-adapter>','exec'),namespace)
    return namespace[node.name]


def make_adapter(aux_receipt):
    def sealed(source):
        context=r.sealed_source(source)
        require(context['role']=='C','dedicated AUX export supports officially loaded C only')
        context['auxiliary_method_identity']=validate_auxiliary_checkpoint(context['checkpoint']['checkpoint'],aux_receipt)
        return context
    def output_name(role,mode,proof):
        require(role=='C','AUX filename requires C')
        return 'PPO_PLUS_LIMITED_AUX_'+r.output_name(role,mode,proof)
    ns={**vars(r),'sealed_source':sealed,'output_name':output_name}
    export=function(OUT/'review_video.py','export',REVIEW_HASH)
    changes={'schema':0,'label':0,'receipt':0}
    for node in ast.walk(export):
        if isinstance(node,ast.Constant) and node.value==OLD_SCHEMA:node.value=SCHEMA;changes['schema']+=1
        elif isinstance(node,ast.Constant) and node.value=='PPO FULL12 CP':node.value='PPO + LIMITED AUX CP';changes['label']+=1
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in node.targets):
            node.value.keys.append(ast.Constant(value='learning_method'))
            node.value.values.append(ast.parse("context['auxiliary_method_identity']",mode='eval').body);changes['receipt']+=1
    require(changes=={'schema':1,'label':1,'receipt':1},'export AST no longer has exactly the reviewed label/receipt sites')
    index=next(i for i,n in enumerate(export.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='count' for t in n.targets))
    export.body.insert(index,ast.parse("label += ' | AUX ' + str(context['auxiliary_method_identity']['accepted_auxiliary_updates_total']) + ' SGD'").body[0])
    aux_export=compiled(export,ns)
    checked=function(OUT/'review_video.py','checked_review',REVIEW_HASH);count=0
    for node in ast.walk(checked):
        if isinstance(node,ast.Constant) and node.value==OLD_SCHEMA:node.value=SCHEMA;count+=1
    require(count==1,'checked-review schema site changed')
    checked.body.insert(-1,ast.parse("require(receipt.get('learning_method') == context['auxiliary_method_identity'], 'AUX media method receipt differs from the actual checkpoint lineage')").body[0])
    aux_checked=compiled(checked,ns)
    q=module('aux_quantity_control_only',OUT/'pair_quantity_only_v2.py')
    quantity=function(OUT/'pair_quantity_only_v2.py','strict_quantity_capture',QUANTITY_HASH)
    guard=ast.dump(ast.parse('reject_auxiliary(infos)').body[0],include_attributes=False)
    removed=[n for n in quantity.body if ast.dump(n,include_attributes=False)==guard]
    require(len(removed)==1,'quantity pure-only gate changed; refuse an unreviewed bypass')
    quantity.body.remove(removed[0])  # Only after independent complete AUX validation below.
    control=compiled(quantity,dict(vars(q)))
    def pair(b_path,c_path,destination):
        b,c=r.checked_review(b_path),aux_checked(c_path)
        require(len(c['receipt']['learning_method']['verified_lineage'])>1,'B comparison requires fresh PPO after the auxiliary boundary')
        common=control(b,c)  # Every runtime/config/hash condition remains unchanged.
        destination=Path(destination).resolve()
        require(destination.is_relative_to(OUT) and destination!=OUT,'use a new isolated AUX pair directory')
        destination.mkdir(parents=True,exist_ok=False)
        encoder=module('aux_encoder_only',ROOT/'outputs/ppo_fl_capture_quality_v1/paired_event_media.py')
        cp=c['receipt']['checkpoint_identity']['saved_global_policy_decisions']
        full=encoder._comparison_video(b,c,destination/f'N_vs_CP{cp}_PPO_PLUS_LIMITED_AUX.mp4',
            left_end=b['receipt']['frame_count'],right_end=c['receipt']['frame_count'],
            kind='full_attempt_same_elapsed_P01_quantity_control_with_explicit_limited_aux_method')
        result={'schema':'wlr50_clean.task_conditioned_limited_aux_control_pair.v1',
            'B_receipt':str(b['receipt_path']),'C_PPO_PLUS_LIMITED_AUX_receipt':str(c['receipt_path']),
            'verified_quantity_only_control_comparison':common,'learning_method':c['receipt']['learning_method'],
            'full_episode':full,'B_physical_result':b['receipt']['physical_result'],
            'C_physical_result':c['receipt']['physical_result'],'quality_improvement_claim':None,
            'task_success_not_inferred_from_aux_or_media':True,'shorter_side_freeze_is_not_new_physics':True}
        r.base.write_new_json(destination/'pair_receipt.json',result);return result
    def modes(det_path,stoch_path,output):
        d,s=aux_checked(det_path),aux_checked(stoch_path)
        common=r.strict_policy_modes(d,s)
        require(d['receipt']['learning_method']==s['receipt']['learning_method'],'different auxiliary lineages in mode pair')
        output=Path(output).resolve();require(output.is_relative_to(OUT),'mode receipt must remain inside outputs')
        result={'schema':'wlr50_clean.task_conditioned_limited_aux_mode_pair.v1',**common,
            'learning_method':d['receipt']['learning_method'],'deterministic_receipt':str(d['receipt_path']),
            'stochastic_receipt':str(s['receipt_path'])}
        r.base.write_new_json(output,result);return result
    return SimpleNamespace(export=aux_export,checked_review=aux_checked,pair=pair,modes=modes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--aux-receipt',type=Path,required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    export=sub.add_parser('export');export.add_argument('--source',type=Path,required=True);export.add_argument('--destination',type=Path,required=True)
    pair=sub.add_parser('pair');pair.add_argument('--b-receipt',type=Path,required=True);pair.add_argument('--c-receipt',type=Path,required=True);pair.add_argument('--destination',type=Path,required=True)
    modes=sub.add_parser('modes');modes.add_argument('--deterministic-receipt',type=Path,required=True);modes.add_argument('--stochastic-receipt',type=Path,required=True);modes.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();adapter=make_adapter(args.aux_receipt)
    result=(adapter.export(args.source,args.destination) if args.command=='export' else
        adapter.pair(args.b_receipt,args.c_receipt,args.destination) if args.command=='pair' else
        adapter.modes(args.deterministic_receipt,args.stochastic_receipt,args.output))
    print(json.dumps(result,indent=2))
